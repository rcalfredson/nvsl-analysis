# standard libraries
import io
import random

# third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import shapely.affinity as sa
import shapely.geometry as sg

# custom modules and constants
from src.utils.common import draw_text_with_bg
from src.utils.constants import _RDP_PKG
from src.analysis.video_analysis_interface import VideoAnalysisInterface

import src.utils.util as util
from src.utils.util import VideoError, COL_BK, COL_O, COL_Y, COL_W

N_TRX_DEFAULT = 24
TRX_IMG_FILE2 = "imgs/%s__t%d_b%d%s.png"


class TrxPlotter:
    """
    A class responsible for plotting and analyzing the trajectories of fruit flies
    based on video analysis data. It supports various plotting modes including grid,
    normalized heatmaps, and regular heatmaps to visualize the movement and behavior
    patterns of the flies.

    Attributes:
        va (VideoAnalysisInterface): An interface to access video analysis data, including
            frames, fly positions, and transformations.
        frame (numpy.ndarray): The current frame from the video analysis data.
        flies (list): A list of flies being analyzed.
        trns (list): A list of transformations or movements recorded for the flies.
        opts (dict): A dictionary of options to customize plotting behavior.
        ntrx (int): The number of trajectories to plot. Defaults to a preset constant.
        bmax (int): The maximum batch number processed. Used internally to manage plotting.
        plot_stats (dict): A dictionary to store statistics about the fly movements, including
            average maximum distance, average first turn angle, and average first run length.

    """

    def __init__(self, va: VideoAnalysisInterface, opts, ntrx=N_TRX_DEFAULT):
        """
        Initializes the TrxPlotter with video analysis data, plotting options, and the
        number of trajectories.

        Parameters:
            va (VideoAnalysisInterface): The video analysis interface with data for plotting.
            opts (dict): Plotting options to customize the output.
            ntrx (int): The number of trajectories to consider for plotting.
        """
        self.va = va
        self.frame = va.frame
        self.flies = va.flies
        self.trns = va.trns
        self.opts = opts
        self.ntrx = ntrx
        self.bmax = 0
        self.plot_stats = {
            "avgMaxDist": [[], []],
            "avgFirstTA": [[], []],
            "avgFirstRL": [[], []],
        }

    def _convert_to_matrix(self, xy):
        """
        Converts a sequence of x and y coordinates into a matrix format suitable for further
        processing and analysis. This method is a utility function primarily used for
        preparing trajectory data before applying transformations such as reduction,
        rotation, and translation.

        Parameters:
            xy (list of tuples or numpy.ndarray): A list of (x, y) tuples or a 2D numpy array
                representing the trajectory coordinates of a single fly.

        Returns:
            numpy.ndarray: A 2D numpy array where each row represents a point in the trajectory
                with its x and y coordinates.
        """
        return util.xy2M(xy)

    def _reduce_points(self, xy):
        """
        Reduces the number of points in a trajectory to simplify its shape while maintaining
        the overall geometry using the Ramer-Douglas-Peucker algorithm. This method is useful
        for minimizing the complexity of trajectories before further processing or analysis.

        Parameters:
            xy (numpy.ndarray): A 2D numpy array representing the trajectory coordinates of
                a single fly, where each row corresponds to a point with x and y coordinates.

        Returns:
            numpy.ndarray: A simplified 2D numpy array of the trajectory with reduced number
                of points, preserving the essential shape and features of the original trajectory.

        Note:
            The degree of simplification is controlled by the `rdp` parameter in the `opts`
            dictionary passed during the initialization of the `TrxPlotter` class. The `_RDP_PKG`
            constant specifies the implementation of the RDP algorithm to be used.
        """
        return util.rdp(xy, self.opts.rdp, _RDP_PKG)

    def _compute_rotation_angle(self, sxy):
        """
        Computes the rotation angle required to align the trajectory with a reference axis.
        This method calculates the angle based on the initial direction of the simplified
        trajectory. The angle is computed such that the trajectory is rotated to align with
        the negative y-axis, facilitating comparison or normalization of directions.

        Parameters:
            sxy (numpy.ndarray): A 2D numpy array representing the simplified trajectory
                coordinates of a single fly, where each row corresponds to a point with x
                and y coordinates. This array is expected to result from the
                `_reduce_points` method.

        Returns:
            float: The rotation angle in radians. If the simplified trajectory has less than
                two points, the angle is set to 0, implying no rotation is necessary.

        Note:
            The rotation angle is calculated using the velocity angles of the trajectory,
            adjusted to align with the standard coordinate system used in mathematical
            and geometric calculations.
        """
        return (
            0
            if len(sxy[0]) < 2
            else util.normAngles(-np.pi / 2 - util.velocityAngles(sxy[0]))[0]
        )

    def _rotate_and_translate(self, xy, ra, orig):
        """
        Applies rotation and translation to the trajectory points to align and position them
        based on the specified rotation angle and origin. This method is used to normalize
        trajectories by aligning them in a common direction and translating them to a common
        origin, which facilitates comparison and analysis of movement patterns.

        Parameters:
            xy (numpy.ndarray): A 2D numpy array representing the original trajectory coordinates
                of a single fly, where each row corresponds to a point with x and y coordinates.
            ra (float): The rotation angle in radians to be applied to the trajectory. This angle
                is typically calculated using the `_compute_rotation_angle` method.
            orig (tuple): A tuple of (x, y) coordinates representing the new origin to which the
                trajectory should be translated after rotation.

        Returns:
            numpy.ndarray: A 2D numpy array of the trajectory coordinates after applying rotation
                and translation, where each row represents a point in the transformed trajectory
                with its new x and y coordinates.

        Note:
            The rotation is performed first, followed by the translation. The transformation is
            applied using geometric operations from the `shapely` library, ensuring accurate and
            efficient processing.
        """
        return np.array(
            sa.rotate(
                sa.translate(sg.LineString(xy), orig[0] - xy[0][0], orig[1] - xy[0][1]),
                ra,
                origin=orig,
                use_radians=True,
            ).coords
        )

    def _convert_to_tuple(self, nxy):
        """
        Converts the normalized trajectory coordinates from a numpy array format into tuples. This method
        is typically used after the trajectory has been processed, for example, after rotation and translation
        transformations, to prepare the data for further analysis or plotting that requires the coordinates
        in separate x and y components.

        Parameters:
            nxy (numpy.ndarray): A 2D numpy array representing the transformed trajectory coordinates
                of a single fly, where each row corresponds to a point with its x and y coordinates
                after normalization processes like rotation and translation.

        Returns:
            tuple: A tuple containing two elements, where the first element is a numpy array of x coordinates
                and the second element is a numpy array of y coordinates of the trajectory points.

        Note:
            This conversion facilitates the use of trajectory data with functions or libraries that
            require x and y coordinates to be provided separately, enhancing compatibility and ease of use.
        """
        return util.xy2T(nxy)

    def _normalize_trajectory(self, xy, orig):
        """
        Normalizes a trajectory by aligning it with a reference direction and translating it to a
        specific origin. The process involves several key steps: converting the trajectory
        coordinates to a matrix format, reducing the number of points using the
        Ramer-Douglas-Peucker algorithm, computing the rotation angle to align with the negative
        y-axis, applying rotation and translation transformations, and finally converting the
        normalized coordinates back into tuples. This comprehensive normalization makes
        trajectories easier to analyze and compare by standardizing their orientation and
        position.

        Parameters:
            xy (list of tuples or numpy.ndarray): The original trajectory coordinates of a single
                                                  fly, either as a list of (x, y) tuples or a 2D
                                                  numpy array.
            orig (tuple): A tuple of (x, y) coordinates representing the new origin to which
                          the trajectory is translated and around which it is rotated.

        Returns:
            tuple: A tuple containing two elements, where the first element is a numpy array
                   of x coordinates and the second element is a numpy array of y coordinates,
                   representing the normalized trajectory points.
        """
        xy = self._convert_to_matrix(xy)
        sxy = self._reduce_points(xy)
        ra = self._compute_rotation_angle(sxy)
        nxy = self._rotate_and_translate(xy, ra, orig)
        return self._convert_to_tuple(nxy)

    def _getSyncBucket(self, t):
        """
        Calculates the synchronization point for a training session and identifies the frame indices
        of all reward events to establish the analysis buckets. This method uses the internal
        VideoAnalysisInterface's _syncBucket method to determine the starting frame index for
        synchronization based on the first reward event and to collect the frame indices for each
        reward event within the session. This enables the segmentation of the training session
        into analyzable parts focused on reward events.

        Args:
            t: The training session to be analyzed, containing details about reward events among
               other data. This session is passed to the VideoAnalysisInterface's _syncBucket method.

        Returns:
            tuple: A tuple containing two elements:
                - fi (int or None): The frame index for the start of the analysis after adjusting for
                                    the first reward event. `None` if no reward events are found.
                - on (list): A list of frame indices for each reward event, used to synchronize the
                             analysis across the session. These indices facilitate focused analysis
                             around reward events by identifying the relevant frames within the video data.
        """
        fi, _, on = self.va._syncBucket(t, self.df, skip=0)
        return fi, on

    def _initialize_plot_properties(self):
        """
        Initializes plot properties specific to the TrxPlotter instance by converting a specified
        bucket length from minutes to frames and extracting the basename of the video file without
        its extension. This method sets up essential plotting parameters that are used in subsequent
        plotting processes, ensuring that the visualization is accurately synchronized with the
        video analysis data.

        This method directly accesses the VideoAnalysisInterface's `_min2f` method to convert the
        `syncBucketLenMin` option (the length of each bucket in minutes) into frames, facilitating
        time-based analysis in terms of video frames. Additionally, it retrieves the basename of
        the video analysis file name for use in labeling or file generation purposes.

        Attributes Set:
            df (int): The frame difference or the length of each bucket in frames, derived from the
                      `syncBucketLenMin` plotting option by converting minutes into the corresponding
                      number of frames based on the video's frame rate.
            fn (str): The basename of the video file being analyzed, obtained without the file extension.
                      This is used for naming conventions in output files or figures generated by the
                      plotting methods.
        """
        self.df, self.fn = self.va._min2f(self.opts.syncBucketLenMin), util.basename(
            self.va.fn, False
        )

    def plotTrx(self, mode="grid"):
        """
        Plots the trajectories of the flies being analyzed in different visualization modes based on the
        specified mode argument. This method serves as a dispatcher, calling different internal methods
        to generate the appropriate type of plot: grid, normalized heatmap, or regular heatmap. Before
        plotting, it initializes plot properties by setting up the frame difference and video file name.

        The trajectory images provide insights into the movement patterns and behavior of the flies
        throughout the video analysis session, aiding in the interpretation of their spatial dynamics
        and interactions.

        Args:
            mode (str, optional): Specifies the plotting mode to use. The available modes are:
                - "grid": Plots individual segments of flies' trajectories around reward events,
                          then arranges these segments in a grid-like formation. This mode visually
                          represents the path of each fly in relation to specific events, allowing
                          for a focused analysis on behavior around these points in time.
                - "hm_norm": Plots a normalized heatmap of the trajectories, emphasizing areas of
                             high frequency movement in a normalized space.
                - "hm": Plots a regular heatmap of the trajectories, highlighting areas of high
                        density movement without normalization.
                The default mode is "grid".

        Raises:
            ValueError: If an unknown plotting mode is specified, an error is raised to inform the
                        user of the invalid mode selection.

        Note:
            This method prints the current mode and a message indicating the start of the trajectory
            image writing process to provide feedback to the user about the ongoing operation.
        """
        print("\nwriting trajectory images...")
        print("mode:", mode)
        self._initialize_plot_properties()
        self.hm = "hm" in mode
        if mode == "grid":
            self._plot_trx_grid()
        elif mode == "hm_norm":
            self._plot_hm_norm()
        elif mode == "hm":
            TrxHmPlotter(self.va, self.trns, self.flies, self.opts, self.fn)._plot_hm()
        else:
            raise ValueError(f"Unknown plotTrx mode: {mode}")

    def _process_trx_batches(
        self, callback, pre_callback=None, post_callback=None, context=None
    ):
        """
        Processes the trajectories in batches, applying specified callback functions at different stages
        of the batch processing. This method iterates through each trajectory segment based on the
        synchronization points and frame intervals determined by the `_getSyncBucket` method. It allows
        for the execution of custom logic before, during, and after the processing of each batch, making
        it versatile for various trajectory analysis tasks.

        Args:
            callback (function): The main callback function to be executed for each segment within a
                                 batch. It is called with the start and end frame indices, the current
                                 training session (`t`), and the context dictionary.
            pre_callback (function, optional): A callback function to be executed before processing
                                               each batch. It is called with the context dictionary.
                                               Default is None.
            post_callback (function, optional): A callback function to be executed after processing
                                                each batch. It is called with the current training
                                                session (`t`), the batch number (`b`), and the context
                                                dictionary. Default is None.
            context (dict, optional): A dictionary for passing and storing arbitrary data between
                                      callbacks and throughout the batch processing lifecycle. If None,
                                      an empty dictionary is initialized and used. Default is None.

        Note:
            This method divides the trajectory data into batches based on the synchronization points
            and frame intervals. Each batch represents a segment of the trajectory data to be analyzed.
            The method dynamically learns the maximum batch number (`bmax`) during the processing of the
            first training session, which is used internally for managing batch sizes and limits.

            The `callback`, `pre_callback`, and `post_callback` functions provide flexibility in
            handling the trajectory data, enabling the implementation of custom analysis and plotting
            logic specific to each phase of the batch processing.
        """
        if context is None:
            context = {}
        for t in self.trns:
            fi, on = self._getSyncBucket(t)
            b = 1
            while fi + self.df <= t.stop:
                if t.n == 1 and self.bmax < b:  # hack: learn bmax in first training
                    self.bmax = b
                f1 = None
                if pre_callback is not None:
                    pre_callback(context)
                for f2 in util.inRange(on, fi, fi + self.df)[: (self.ntrx + 1)]:
                    if f1:
                        callback(f1, f2, t, context)
                    f1 = f2
                if post_callback is not None:
                    post_callback(t, b, context)
                b += 1
                fi += self.df

    def _plot_trx_grid(self):
        """
        Plots the trajectories of flies in a grid format by segmenting their movements around
        reward events and arranging these segments in a visually structured manner. This method
        initializes a context with the frame index of the first reward event and utilizes the
        `_process_trx_batches` method to systematically generate and process each trajectory
        segment.

        The method employs specific callback functions to handle different stages of the plotting
        process:
        - A pre-callback function to initialize the context for each batch of trajectory segments.
        - A main callback function to draw each segment of the trajectory on individual frames.
        - A post-callback function to compile the drawn segments into a grid and save the resulting image.

        This structured approach allows for detailed visualization of fly behavior in relation to
        reward events, providing insights into movement patterns and interactions within the training
        session.

        Note:
            The context dictionary is used to pass the starting frame index (`f0`) of the analysis,
            which is based on the synchronization point determined by the `_getSyncBucket` method.
            This starting point is essential for aligning the trajectory segments with the timing
            of reward events.
        """
        context = {"f0": self._getSyncBucket(self.trns[0])[0]}
        self._process_trx_batches(
            self._plot_grid_callback,
            pre_callback=self._plot_grid_pre_callback,
            post_callback=self._plot_grid_post_callback,
            context=context,
        )

    def _plot_grid_callback(self, f1, f2, t, context):
        """
        Callback function used by `_plot_trx_grid` to plot and annotate a single segment of a
        fly's trajectory between two frame indices. This function is responsible for generating
        a visual representation of each trajectory segment, adding it to a collection of images
        to be arranged in a grid format.

        The function reads the video frame corresponding to the end of the segment, draws the
        trajectory lines and points on this frame, and optionally applies a Ramer-Douglas-Peucker
        simplification if specified in the options. It also calculates and displays the first turn
        angle for each segment, providing additional behavioral insights.

        Args:
            f1 (int): The starting frame index of the current trajectory segment.
            f2 (int): The ending frame index of the current trajectory segment.
            t (TrainingSession): The current training session object, containing data related to the
                                 session being processed.
            context (dict): A dictionary containing context-specific data needed for processing,
                            including the initial frame index (`f0`), and lists for storing images
                            (`imgs`) and headers (`hdrs`).

        Note:
            This method updates the `imgs` and `hdrs` lists in the context with the newly created
            image and its corresponding header, which includes time stamps and frame ranges. The
            trajectory is drawn using specified colors for the trajectory lines and points, with
            additional visual cues for simplified points and turn angles when applicable.
        """
        f0 = context["f0"]
        imgs = context["imgs"]
        hdrs = context["hdrs"]
        try:
            img = util.readFrame(self.va.cap, f2)
        except VideoError:
            print("could not read frame %d" % f2)
            img = self.frame.copy()
            pass
        t.annotate(img, col=COL_BK)
        txt = []
        for f in self.flies:
            trx = self.va.trx[f]
            xy = trx.xy(f1, f2 + 1)
            pts = util.xy2Pts(*xy)

            cv2.polylines(img, pts, False, COL_W)
            cv2.circle(img, tuple(pts[0, -1, :]), 3, COL_W, -1)
            if self.opts.rdp:
                sxy = trx.xyRdp(f1, f2 + 1, epsilon=self.opts.rdp)
                spts = util.xy2Pts(*(sxy[0]))
                cv2.polylines(img, spts, False, COL_Y)
                for i in range(1, spts.shape[1] - 1):
                    cv2.circle(img, tuple(spts[0, i, :]), 2, COL_Y, -1)
                tas = util.turnAngles(sxy[0])
                txt.append(
                    "ta0 = %s" % ("%.1f" % (tas[0] * 180 / np.pi) if len(tas) else "NA")
                )
        if txt:
            util.putText(
                img,
                ", ".join(txt),
                (5, 5),
                (0, 1),
                util.textStyle(color=COL_W),
            )
        imgs.append(img)
        hdrs.append("%s (%d-%d)" % (self.va._f2ms(f2 - f0), f1, f2))

    def _plot_grid_pre_callback(self, context):
        """
        Pre-callback function for initializing the context with empty lists for images and headers
        before processing each batch of trajectory segments in the grid plotting mode. This method
        prepares the context for a new set of trajectory segment images and their corresponding headers
        to be generated and stored during the batch processing.

        The `context` dictionary is updated in place, adding or resetting the `imgs` and `hdrs` keys
        to empty lists. These lists will then be populated with the images of trajectory segments and
        their headers by the main callback function as each segment is processed.

        Args:
            context (dict): A dictionary containing context-specific data needed for processing
                            trajectory segments. This dictionary is updated with empty lists for
                            `imgs` and `hdrs`, which are used to store the generated images and
                            their headers, respectively.

        Note:
            This function is specifically designed to be used as a pre-callback in the
            `_process_trx_batches` method when plotting trajectories in a grid format. It ensures
            that each batch starts with a clean slate for storing images and headers, thereby
            facilitating the organized accumulation and subsequent compilation of trajectory
            segment visualizations.
        """
        context["imgs"] = []
        context["hdrs"] = []

    def _plot_grid_post_callback(self, t, b, context):
        """
        Post-callback function for finalizing and saving the grid of trajectory segment images
        after processing each batch in the grid plotting mode. This method compiles the generated
        images of trajectory segments into a single image arranged in a grid format and saves this
        image to a file.

        It retrieves the list of images and their corresponding headers from the context, combines
        them into one image with a specified number of columns (nc), and then writes the resulting
        grid image to disk using a predefined naming convention that includes the video file name,
        the training session number, and the batch number.

        Args:
            t (TrainingSession): The current training session object, containing data related to the
                                 session being processed. Used here for incorporating the session
                                 number into the filename.
            b (int): The batch number, indicating the current batch of trajectory segments being
                     processed. It's used in the filename to differentiate between batches.
            context (dict): A dictionary containing context-specific data needed for processing,
                            including lists of images (`imgs`) and headers (`hdrs`) generated during
                            the batch processing.

        Note:
            The filename for the saved image is constructed using a format string that incorporates
            the video file basename (`fn`), the training session number (`t.n`), and the batch number
            (`b`), ensuring that each batch's grid image is uniquely identifiable. The `nc` parameter
            for `util.combineImgs` determines the number of columns in the grid, affecting the layout
            of the compiled image.
        """
        imgs = context["imgs"]
        hdrs = context["hdrs"]
        img = util.combineImgs(imgs, hdrs=hdrs, nc=6)[0]
        cv2.imwrite(TRX_IMG_FILE2 % (self.fn, t.n, b, ""), img)

    def _plot_hm_norm(self):
        """
        Initiates the plotting of normalized heatmaps for fly trajectories, visualizing movement
        patterns with respect to a normalized center point for each fly. This method sets up the
        initial context for heatmap generation, including the dimensions of the heatmap and a
        function to calculate the center point for normalization.

        The method then processes trajectory batches through specified callback functions, which
        collectively contribute to the generation of a comprehensive heatmap that emphasizes areas
        of frequent activity and movement patterns of the flies within the analyzed video frames.
        """
        w, h = util.imgSize(self.frame)
        img1 = util.getImg(2 * h, 2 * w, 1, 0)

        def center(f):
            return util.intR((0.5 + f) * w, h)

        context = {"w": w, "h": h, "img1": img1, "center": center}
        self._process_trx_batches(
            self._plot_hm_norm_callback,
            self._plot_hm_norm_pre_callback,
            self._plot_hm_norm_post_callback,
            context,
        )

    def _plot_hm_norm_callback(self, f1, f2, t, context):
        """
        Callback function for processing each trajectory segment during the generation of normalized
        heatmaps. This function calculates and updates statistics such as maximum distances, turn
        angles, and run lengths for each fly, and plots the normalized trajectory segments onto an
        intermediate heatmap.

        Args:
            f1, f2 (int): Frame indices defining the current trajectory segment.
            t (TrainingSession): The current training session object.
            context (dict): Context data including the intermediate heatmap (`mp`), a function for
                            calculating the center (`center`), and lists for accumulating statistics
                            (`maxDs`, `turnAs`, `runLs`).
        """
        mp = context["mp"]
        center = context["center"]
        maxDs = context["maxDs"]
        turnAs = context["turnAs"]
        runLs = context["runLs"]
        img1 = context["img1"]
        for f in self.flies:
            trx = self.va.trx[f]
            xy = trx.xy(f1, f2 + 1)
            maxDs[f].append(np.max(util.distances(xy, True)))
            sxy = trx.xyRdp(f1, f2 + 1, epsilon=self.opts.rdp)
            tas = util.turnAngles(sxy[0])
            rls = util.distances(sxy[0])
            if not (len(tas) or len(rls)):
                continue
            if len(tas):
                turnAs[f].append(tas[0])
            if len(rls):
                runLs[f].append(rls[0])
            xy = self._normalize_trajectory(xy, center(f))
            img1[...] = 0
            cv2.polylines(img1, util.xy2Pts(*xy), False, 1)
            mp += img1

    def _plot_hm_norm_pre_callback(self, context):
        """
        Pre-callback function for setting up the context with initial values and structures necessary
        for generating normalized heatmaps. Initializes the main heatmap matrix and lists for storing
        statistical measures of movement patterns such as maximum distances, turn angles, and run
        lengths.

        Args:
            context (dict): A dictionary to be populated with initial structures for heatmap generation.
        """
        h = context["h"]
        w = context["w"]
        context["mp"] = np.ones((2 * h, 2 * w), np.float32)
        for k in ("maxDs", "turnAs", "runLs"):
            context[k] = [[], []]

    def _plot_hm_norm_post_callback(self, t, b, context):
        """
        Post-callback function for finalizing the normalized heatmap after processing all trajectory
        segments. This function applies a colormap to the heatmap, marks significant features such as
        average maximum distance and centers of mass, and saves the resulting image to a file.

        Args:
            t (TrainingSession): The current training session object.
            b (int): The batch number, indicating the current batch of trajectory segments.
            context (dict): Context data containing the accumulated heatmap (`mp`) and statistical
                            measures (`maxDs`, `turnAs`, `runLs`).
        """
        center = context["center"]
        mp = context["mp"]
        turnAs = context["turnAs"]
        runLs = context["runLs"]
        h = context["h"]
        img = util.heatmap(mp, colormap=cv2.COLORMAP_VIRIDIS)
        for f in self.flies:
            # start circle
            c = center(f)
            cv2.circle(img, c, 3, COL_W, -1)
            # average max distance
            amd = np.mean(context["maxDs"][f])
            r = util.intR(amd)
            cv2.circle(img, c, r, COL_W)
            # center of mass (inside average max distance)
            mp1 = mp - 1
            msk = np.zeros_like(mp1, dtype=np.uint8)
            cv2.circle(msk, c, r, 1, -1)
            mp1[msk == 0] = 0
            com = ndi.center_of_mass(mp1)
            # for debugging:
            # print msk[h-5:h+5,f*w+w/2-5:f*w+w/2+5]
            cv2.circle(img, util.intR(com[::-1]), 3, COL_O, -1)
            # turn angles and run lengths
            atad = arl = None
            if turnAs[f] and runLs[f]:
                ata, arl = np.mean(np.abs(turnAs[f])), np.mean(runLs[f])
                atad = ata * 180 / np.pi
                c = util.tupleAdd(c, (0, h / 2))
                cv2.line(
                    img,
                    util.intR(c),
                    util.intR(c[0] + arl * np.sin(ata), c[1] - arl * np.cos(ata)),
                    COL_W,
                )
            if b <= self.bmax:
                self.plot_stats["avgMaxDist"][f].append(amd)
                self.plot_stats["avgFirstTA"][f].append(atad)
                self.plot_stats["avgFirstRL"][f].append(arl)
        cv2.imwrite(TRX_IMG_FILE2 % (self.fn, t.n, b, "_hm"), img)


class TrxHmPlotter:
    PADDING = 10

    def __init__(self, va, trns, flies, opts, fn):
        """
        Initialize the TrxHmPlotter class.

        Parameters:
        - va: Video analysis object.
        - trns: List of training events.
        - flies: List of fly indices to analyze.
        - opts: Options for generating the heatmap.
        - fn: Filename for saving the heatmap.
        """
        self.va = va
        self.trns = trns
        self.flies = flies
        self.opts = opts
        self.fn = fn
        self.target_width = 800
        if opts.groupLabels is not None:
            self.gl = opts.groupLabels.split("|")[va.gidx]
        else:
            self.gl = None

        self.trajectories = None
        self.fly_num_absolute = None
        self.fly_type = None
        self.upper_caption = None
        self.rwd_circle = [None, None]
        self.bounds = [None] * 4
        self.avg_frame_cropped = None
        self.w_cropped = None
        self.h_cropped = None
        self.aspect_ratio = None
        self.scale_factor = None
        self.frame_rgb = None
        self.avg_frame = None

    def _get_frame(self, index=1000):
        """
        Retrieve a frame from the video at a given index.

        Parameters:
        - index: Integer index of the frame to retrieve. Default is 1000.

        Returns:
        - frame: The retrieved frame from the video.
        """
        try:
            frame = util.readFrame(self.va.cap, index)
        except VideoError:
            print(f"could not read frame {index}")
            frame = self.va.frame.copy()
        return frame

    def _get_average_frame(self, num_frames=10):
        """
        Retrieve an average frame from the video using `num_frames` random frames.

        Parameters:
        - num_frames: Number of random frames to average. Default is 10.

        Returns:
        - avg_frame: The averaged frame from the video.
        """
        total_frames = int(self.va.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frames = random.sample(range(total_frames), num_frames)

        # Initialize an accumulator for frames
        accumulator = None
        for idx in sampled_frames:
            frame = self._get_frame(idx)
            if accumulator is None:
                accumulator = np.zeros_like(frame, dtype=np.float64)
            accumulator += frame

        # Calculate the average of the frames
        avg_frame = (accumulator / num_frames).astype(np.uint8)
        return avg_frame

    def _compute_raw_heatmap(self, trajectories):
        """
        Compute a raw heatmap from the given trajectories and bounds.

        Parameters:
        - trajectories: List of trajectories to generate the heatmap.

        Returns:
        - heatmap_img: The generated raw heatmap image.
        """
        xm, ym, xM, yM = self.bounds
        heatmap_img = np.zeros((yM - ym, xM - xm), dtype=np.uint8)

        for trj in trajectories:
            x, y = trj[0], trj[1]
            heatmap_tmp = np.zeros_like(heatmap_img)
            pts = util.xy2Pts(x, y)
            cv2.polylines(heatmap_tmp, pts, False, 1)
            heatmap_img += heatmap_tmp
        return heatmap_img

    def _translate_trajectories(self, shift_x, shift_y):
        """
        Translate trajectories based on given x and y shifts.

        Parameters:
        - shift_x (int or float): Amount to shift in the x direction.
        - shift_y (int or float): Amount to shift in the y direction.
        """
        translated_trajectories = []
        for trj in self.trajectories:
            translated_trajectories.append(
                list(
                    self.va.xf.f2t(
                        trj[:, 0], trj[:, 1], f=self.fly_num_absolute, noMirr=True
                    )
                )
            )
            translated_trajectories[-1][0] -= shift_x / self.va.xf.fctr
            translated_trajectories[-1][1] -= shift_y / self.va.xf.fctr
        self.trajectories = translated_trajectories

    def _initialize_distance_storage(self):
        """
        Initialize the storage for distances with NaN placeholders for each segment of trajectories.
        """
        if not hasattr(self.va, "btwnRwdDistsFromCtr"):
            self.va.btwnRwdDistsFromCtr = [
                [
                    [float("nan"), float("nan"), float("nan"), float("nan")]
                    for _ in range(len(self.flies))
                ]
                for _ in range(len(self.trns))
            ]

    def _calculate_avg_distances(self, trajectories, training_num, segment_index):
        """
        Calculate average max and mean distances for a list of trajectories.
        Updates the VideoAnalysis instance with these statistics.

        Parameters:
        - trajectories: List of trajectories to analyze.
        - training_num: Number indicating the training event.
        - segment_index: Index to distinguish first 24 (0) and last 24 (2) trajectories.

        Returns:
        - Tuple of average max and mean distances for the given trajectories.
        """
        mean_max_distances = []
        mean_mean_distances = []
        for trj in trajectories:
            x, y = trj[0], trj[1]
            distances = np.sqrt(
                (x - self.rwd_circle[0]) ** 2 + (y - self.rwd_circle[1]) ** 2
            )
            mean_max_distances.append(np.nanmax(distances))
            mean_mean_distances.append(np.nanmean(distances))

        avg_max_distance = np.mean(mean_max_distances)
        avg_mean_distance = np.mean(mean_mean_distances)
        fly_idx = 0 if self.fly_type == "Experimental" else 1

        # Update the data structure accordingly
        if training_num is not None:
            self.va.btwnRwdDistsFromCtr[training_num][fly_idx][
                segment_index
            ] = avg_max_distance
            self.va.btwnRwdDistsFromCtr[training_num][fly_idx][
                segment_index + 1
            ] = avg_mean_distance

        return avg_max_distance, avg_mean_distance

    def _draw_circle_and_radius(self, image, distance, center, scale_factor, color):
        """
        Draw a circle with labeled radius on an image.

        Parameters:
        - image: Image to draw upon.
        - distance: Radius of the circle to draw.
        - center: Center coordinates of the circle.
        - scale_factor: Factor to scale the circle by.
        - color: Color of the circle and text.
        """
        center = (int(center[0] * scale_factor), int(center[1] * scale_factor))
        cv2.circle(
            image,
            center,
            int(distance * scale_factor),
            color,
            thickness=3,
        )
        draw_text_with_bg(
            image,
            f"R: {round(distance / self.va.ct.pxPerMmFloor(), 2)} mm",
            (
                center[0] + 10,
                center[1] - int(distance * scale_factor) - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COL_BK,
            1,
            cv2.LINE_AA,
            COL_W,
        )

    def _add_colorbar_to_image(self, img, colormap, vmin, vmax, caption="Counts"):
        """
        Add a color scale bar to the right side of the image with a caption.

        Parameters:
        - img: Image to which the colorbar needs to be added.
        - colormap: Colormap to be used for the colorbar (e.g., "jet").
        - vmin: Minimum value of the data.
        - vmax: Maximum value of the data.
        - caption: Caption for the colorbar.

        Returns:
        - final_img_with_cbar: Image with the colorbar added.
        """
        fig, ax = plt.subplots(
            figsize=(img.shape[1] / 100 / 7, img.shape[0] / 100), dpi=100
        )
        fig.subplots_adjust(0, 0, 1, 1)
        ax.axis("off")
        cax = fig.add_axes([0.3, 0.1, 0.3, 0.8])
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin, vmax)),
            cax=cax,
        )

        cbar.ax.set_title(caption, rotation=0, va="bottom", ha="center")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        cbar_img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), 1)
        buf.close()
        plt.close()

        final_img_with_cbar = np.hstack((img, cbar_img))

        return final_img_with_cbar

    def _generate_filename(self, training_num, ef, postfix):
        """
        Generate a filename based on given parameters.

        Parameters:
        - training_num: Number of the training event.
        - ef: Index of the exp/yoked pair of flies.
        - postfix: Additional string to append to the filename.

        Returns:
        - Filename string.
        """
        trx_img = "imgs/%s__t%d_f%d_%s_%s.png"
        return trx_img % (
            self.fn,
            training_num + 1,
            ef,
            self.fly_type[:3].lower(),
            postfix,
        )

    def _resize_image(self, img, aspect_ratio, interpolation=cv2.INTER_CUBIC):
        """
        Resize an image while maintaining a specified aspect ratio.

        Parameters:
        - img: Image to be resized.
        - aspect_ratio: Aspect ratio to maintain during resizing.
        - interpolation: Interpolation method used in resizing. Default is cv2.INTER_CUBIC.

        Returns:
        - Resized image.
        """
        return cv2.resize(
            img,
            (self.target_width, int(self.target_width * aspect_ratio)),
            interpolation=interpolation,
        )

    def generate_heatmap(self, trajectories, caption, training_num, segment_index):
        """
        Generate a heatmap based on given trajectories, caption, and bounds.

        Parameters:
        - trajectories: List of trajectories to generate the heatmap.
        - caption: Caption to display on the generated heatmap.
        - training_num: Number indicating the training event.
        - segment_index: Index to distinguish first 24 (0) and last 24 (2) trajectories.

        Returns:
        - final_image: The final heatmap image.
        """
        heatmap_image = self._compute_raw_heatmap(trajectories)
        heatmap_rgb, alpha_mask_resized = self._prepare_heatmap_images(heatmap_image)
        blended_rgb = self._blend_images(heatmap_rgb, alpha_mask_resized)
        final_image = self._annotate_heatmap(
            blended_rgb, trajectories, caption, training_num, segment_index
        )
        final_image = self._add_final_touches(final_image, heatmap_image)
        return final_image

    def _prepare_heatmap_images(self, heatmap_image):
        """
        Prepare the heatmap image and its alpha mask for blending.

        Parameters:
        - heatmap_image: Original heatmap image.

        Returns:
        - heatmap_rgb: Resized and normalized heatmap image.
        - alpha_mask_resized: Resized alpha mask of the heatmap.
        """
        mask = heatmap_image > 0
        alpha_mask = np.where(mask, 1.0, 0.3).astype(np.float32)
        heatmap_image_colored = util.heatmap(
            heatmap_image, xform=None, colormap=cv2.COLORMAP_VIRIDIS
        )

        alpha_mask_resized = self._resize_image(
            alpha_mask, self.aspect_ratio, cv2.INTER_NEAREST
        )
        heatmap_rgb = self._resize_image(
            heatmap_image_colored, self.aspect_ratio, cv2.INTER_NEAREST
        )

        heatmap_rgb = heatmap_rgb.astype(np.float32) / 255.0
        return heatmap_rgb, alpha_mask_resized

    def _blend_images(self, heatmap_rgb, alpha_mask_resized):
        """
        Blend the heatmap image with the frame using the alpha mask.

        Parameters:
        - heatmap_rgb: Resized and normalized heatmap image.
        - alpha_mask_resized: Resized alpha mask of the heatmap.

        Returns:
        - blended_rgb: Blended heatmap image.
        """
        blended_rgb = (
            1 - alpha_mask_resized[:, :, np.newaxis]
        ) * self.frame_rgb + alpha_mask_resized[:, :, np.newaxis] * heatmap_rgb
        return (blended_rgb * 255).astype(np.uint8)

    def _annotate_heatmap(
        self, blended_rgb, trajectories, caption, training_num, segment_index
    ):
        """
        Annotate the heatmap with circles representing average distances
        and a caption.

        Parameters:
        - blended_rgb: The blended heatmap image.
        - trajectories: List of trajectories used in heatmap generation.
        - caption: Caption to display on the heatmap.
        - training_num: Number indicating the training event.
        - segment_index: Index to distinguish first 24 (0) and last 24 (2) trajectories.

        Returns:
        - final_image: Annotated heatmap image.
        """
        final_image = np.ascontiguousarray(blended_rgb)
        avg_max_distance, avg_mean_distance = self._calculate_avg_distances(
            trajectories, training_num, segment_index
        )

        self._draw_circle_and_radius(
            final_image,
            self.rwd_circle[2],
            self.rwd_circle[:2],
            self.scale_factor,
            COL_BK,
        )
        self._draw_circle_and_radius(
            final_image, avg_max_distance, self.rwd_circle, self.scale_factor, COL_W
        )
        self._draw_circle_and_radius(
            final_image, avg_mean_distance, self.rwd_circle, self.scale_factor, COL_Y
        )

        draw_text_with_bg(
            final_image,
            caption,
            (20, final_image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            COL_BK,
            2,
            cv2.LINE_AA,
            COL_W,
        )
        return final_image

    def _add_final_touches(self, final_image, heatmap_img):
        """
        Add final touches to the heatmap, including borders, upper caption,
        and a colorbar.

        Parameters:
        - final_image: The annotated heatmap image.
        - heatmap_img: Original heatmap image used for determining colorbar bounds.

        Returns:
        - colorbar_img: Heatmap image with added colorbar.
        """
        border_thickness = 40
        final_image = cv2.copyMakeBorder(
            final_image,
            border_thickness,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        vmin, vmax = np.min(heatmap_img), np.max(heatmap_img)
        colorbar_img = self._add_colorbar_to_image(final_image, "viridis", vmin, vmax)
        cv2.putText(
            colorbar_img,
            self.upper_caption,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return colorbar_img

    def _save_heatmap(
        self, trajectories, description, training_num, postfix, segment_index
    ):
        """
        Save a heatmap image generated from the provided trajectories to disk.

        Parameters:
        - trajectories: List of trajectories from which the heatmap is generated.
        - description: Text description/caption to add to the heatmap image.
        - training_num: Number indicating the training event.
        - postfix: Additional string to append to the end of the filename.
        - segment_index: Index to distinguish first 24 (0) and last 24 (2) trajectories.
        """
        heatmap = self.generate_heatmap(
            trajectories, description, training_num, segment_index
        )
        filename = self._generate_filename(training_num, self.va.ef, postfix)
        cv2.imwrite(filename, heatmap)

    def _setup_plot(self, f):
        """
        Set up the required parameters and data for the plotting process based on a given fly index.

        Parameters:
        - f: Index of the fly for which the heatmap needs to be generated.
        """
        self.fly_num_absolute = self.va.ef + (f * self.va.nef)
        self.fly_type = "Experimental" if f == 0 else "Yoked"
        video_split = self.fn.split("__")
        self.upper_caption = (
            f"{video_split[0]}, fly {self.va.ef} |"
            f" Rec. start: {video_split[2].replace('-', ':')}, {video_split[1]}"
        )
        if self.gl is not None:
            self.upper_caption += f" | Group: {self.gl}"

    def _calculate_bounds(self, f):
        """
        Calculate the default bounds for the heatmap.

        Parameters:
        - f: Index of the fly.

        Returns:
        - Tuple containing minimum and maximum coordinates.
        """
        return self.va.ct.floor(self.va.xf, f=self.va.ef + (f * self.va.nef))

    def _add_padding(self, min_coords, max_coords):
        """
        Add padding to the calculated bounds.

        Parameters:
        - min_coords: Tuple containing minimum x and y coordinates.
        - max_coords: Tuple containing maximum x and y coordinates.

        Returns:
        - Tuple with padded bounds.
        """
        min_x, min_y = min_coords
        max_x, max_y = max_coords
        return (
            max(min_x - self.PADDING, 0),
            max(min_y - self.PADDING, 0),
            max_x + self.PADDING,
            max_y + self.PADDING,
        )

    def _calculate_shift(self, original_min_coords):
        """
        Calculate the shift based on the padded bounds.

        Parameters:
        - original_min_coords: Original minimum coordinates.

        Returns:
        - Tuple containing shift in x and y directions.
        """
        original_min_x, original_min_y = original_min_coords
        return self.bounds[0] - original_min_x, self.bounds[1] - original_min_y

    def _adjust_circle_position(self, shift_x, shift_y):
        """
        Adjust the position of the circle based on the shift.

        Parameters:
        - shift_x: Shift in the x direction.
        - shift_y: Shift in the y direction.
        """
        self.rwd_circle[0] -= shift_x / self.va.xf.fctr
        self.rwd_circle[1] -= shift_y / self.va.xf.fctr

    def _update_frame_attributes(self):
        """
        Update frame-related attributes based on the adjusted bounds.
        """
        self.avg_frame_cropped = self.avg_frame[
            self.bounds[1] : self.bounds[3], self.bounds[0] : self.bounds[2]
        ]
        self.w_cropped = self.bounds[2] - self.bounds[0]
        self.h_cropped = self.bounds[3] - self.bounds[1]
        self.aspect_ratio = self.h_cropped / self.w_cropped
        self.scale_factor = self.target_width / self.w_cropped
        self.frame_rgb = self._resize_image(self.avg_frame_cropped, self.aspect_ratio)
        self.frame_rgb = self.frame_rgb.astype(np.float32) / 255.0

    def _adjust_bounds(self, f):
        """
        Adjust the bounds of the heatmap and related attributes.

        Parameters:
        - f: Index of the fly for which the heatmap needs to be generated.
        """
        min_coords, max_coords = self._calculate_bounds(f)
        self.bounds = self._add_padding(min_coords, max_coords)
        shift_x, shift_y = self._calculate_shift(min_coords)

        self._translate_trajectories(shift_x, shift_y)
        self._adjust_circle_position(shift_x, shift_y)
        self._update_frame_attributes()

    def _plot_hm(self):
        """
        Process and plot heatmaps for specified trainings, flies, and events.
        """
        self.avg_frame = self._get_average_frame()
        self._initialize_distance_storage()
        for t in self.trns:
            for f in self.flies:
                if self.va._bad(f):
                    continue
                self._setup_plot(f)
                fetch_events = lambda t, f=f: self.va._getOn(
                    t, calc=bool(f), f=f if f else None
                )
                events = fetch_events(t)
                self.trajectories = []
                for i in range(len(events) - 1):
                    if events[i] == events[i + 1]:
                        continue
                    trj = self.va.trx[f].xyRdp(
                        events[i], events[i + 1], epsilon=self.opts.rdp
                    )[0]

                    self.trajectories.append(trj)
                if len(self.trajectories) < 48:
                    continue
                circle = t.circles(f)[0]
                self.rwd_circle = list(
                    self.va.xf.f2t(circle[0], circle[1], f=self.fly_num_absolute)
                ) + [circle[2]]
                self._adjust_bounds(f)

                heatmap_info = [
                    (self.trajectories[:24], "First 24", "_hm_first_24"),
                    (self.trajectories[-24:], "Last 24", "_hm_last_24"),
                ]

                for i, (trajectory, title_part, file_tag) in enumerate(heatmap_info):
                    self._save_heatmap(
                        trajectory,
                        f"{self.fly_type} | Training {t.n} | {title_part}",
                        t.n - 1,
                        file_tag,
                        i * 2,
                    )
            self.va.btwnRwdDistsFromCtr[t.n - 1] = util.concat(
                self.va.btwnRwdDistsFromCtr[t.n - 1]
            )
