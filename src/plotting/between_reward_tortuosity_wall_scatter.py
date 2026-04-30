from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Sequence

import numpy as np

from src.plotting.between_reward_segment_binning import sync_bucket_window
from src.plotting.between_reward_segment_metrics import (
    seg_keep_frames,
    tortuosity_metric_masked,
)
from src.plotting.wall_contact_utils import build_wall_contact_mask_for_window
from src.utils import util


@dataclass
class BetweenRewardTortuosityWallScatterConfig:
    export_npz: str | None = None
    training_index: int = 0
    skip_first_sync_buckets: int = 0
    keep_first_sync_buckets: int = 0
    metric_mode: str = "path_over_max_radius"
    segment_scope: str = "full"  # "full" or "return_leg"
    exclude_nonwalking_frames: bool = False
    exclude_reward_endpoints: bool = False
    min_walk_frames: int = 2
    min_displacement_mm: float = 0.0
    min_radius_mm: float = 0.0


class BetweenRewardTortuosityWallScatterExporter:
    """
    Export segment-level tortuosity versus wall-contact fraction.

    Wall contact is measured in the same full-path or return-leg metric window
    used for tortuosity, but wall-contact segments are not excluded.
    """

    def __init__(
        self,
        vas: Sequence["VideoAnalysis"],
        opts,
        gls,
        cfg: BetweenRewardTortuosityWallScatterConfig,
    ):
        self.vas = vas
        self.opts = opts
        self.gls = gls
        self.cfg = cfg
        self.log_tag = "btw_rwd_tortuosity_wall_scatter"

    @staticmethod
    def _normalize_segment_scope(scope: str | None) -> str:
        scope = str(scope or "full").strip().lower().replace("-", "_")
        if scope in {"return", "return_leg", "tail"}:
            return "return_leg"
        return "full"

    @staticmethod
    def _unit_id(va, *, f: int) -> str:
        return f"{getattr(va, 'fn', 'unknown_video')}|trx_idx={int(f)}"

    @staticmethod
    def _video_base(va) -> str:
        fn = getattr(va, "fn", None)
        if fn:
            try:
                return os.path.splitext(os.path.basename(str(fn)))[0]
            except Exception:
                pass
        return f"va_{id(va)}"

    @staticmethod
    def _fly_id(va, *, role_idx: int, trx_idx: int) -> int:
        f = getattr(va, "f", None)
        if f is None:
            return -1
        try:
            if isinstance(f, (list, tuple, np.ndarray)):
                if len(f) > role_idx:
                    return int(f[role_idx])
                if len(f) > trx_idx:
                    return int(f[trx_idx])
        except Exception:
            pass
        try:
            return int(f)
        except Exception:
            return -1

    @staticmethod
    def _build_nonwalk_mask(va, f: int, *, fi: int, n_frames: int, enabled: bool):
        if not enabled:
            return None
        traj = va.trx[f]
        walking = getattr(traj, "walking", None)
        if walking is None:
            return None
        s0 = max(0, min(int(fi), len(walking)))
        e0 = max(0, min(int(fi + n_frames), len(walking)))
        wwin = np.zeros((int(max(1, n_frames)),), dtype=bool)
        if e0 > s0:
            wseg = np.asarray(walking[s0:e0], dtype=float)
            wseg = np.where(np.isfinite(wseg), wseg, 0.0)
            wwin[: len(wseg)] = wseg > 0
        return ~wwin

    def _segment_wall_fraction(
        self,
        *,
        traj,
        s: int,
        e: int,
        fi: int,
        wc: np.ndarray,
        nonwalk_mask,
        exclude_nonwalk: bool,
        min_walk_frames: int,
    ) -> tuple[float, int, int]:
        keep, L = seg_keep_frames(
            traj=traj,
            s=s,
            e=e,
            fi=fi,
            nonwalk_mask=nonwalk_mask,
            exclude_nonwalk=exclude_nonwalk,
            min_keep_frames=min_walk_frames,
        )
        if keep is None or L <= 0:
            return np.nan, 0, 0
        s2 = max(0, min(int(s - fi), len(wc)))
        e2 = max(0, min(int(s - fi + L), len(wc)))
        if e2 <= s2:
            return np.nan, 0, int(np.sum(keep))
        wseg = np.asarray(wc[s2:e2], dtype=bool)
        n = int(min(wseg.size, keep.size))
        if n <= 0:
            return np.nan, 0, int(np.sum(keep))
        keep = keep[:n]
        wseg = wseg[:n]
        denom = int(np.sum(keep))
        if denom <= 0:
            return np.nan, 0, 0
        numer = int(np.sum(wseg[keep]))
        return float(numer / denom), numer, denom

    def collect_records(self) -> dict[str, np.ndarray | dict]:
        t_idx = int(max(0, self.cfg.training_index))
        scope = self._normalize_segment_scope(self.cfg.segment_scope)
        needs_max_frame = scope == "return_leg"
        warned_missing_wc = [False]

        rows = []
        for va in self.vas:
            if getattr(va, "_skipped", False):
                continue
            if getattr(va, "trx", None) is None or len(va.trx) == 0:
                continue
            if va.trx[0].bad() or t_idx >= len(getattr(va, "trns", [])):
                continue

            trn = va.trns[t_idx]
            video_id = self._video_base(va)
            for role_idx, f in enumerate(va.flies):
                if not va.noyc and f != 0:
                    continue

                traj = va.trx[f]
                fi, df, n_buckets, complete = sync_bucket_window(
                    va,
                    trn,
                    t_idx=t_idx,
                    f=f,
                    skip_first=int(max(0, self.cfg.skip_first_sync_buckets or 0)),
                    keep_first=int(max(0, self.cfg.keep_first_sync_buckets or 0)),
                    use_exclusion_mask=False,
                )
                if n_buckets <= 0:
                    continue

                n_frames = int(max(1, n_buckets * df))
                wc = build_wall_contact_mask_for_window(
                    va,
                    f,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=True,
                    warned_missing_wc=warned_missing_wc,
                    log_tag=self.log_tag,
                )
                if wc is None:
                    continue

                nonwalk_mask = self._build_nonwalk_mask(
                    va,
                    f,
                    fi=fi,
                    n_frames=n_frames,
                    enabled=bool(self.cfg.exclude_nonwalking_frames),
                )
                px_per_mm = float(traj.pxPerMmFloor * va.xf.fctr)
                if not np.isfinite(px_per_mm) or px_per_mm <= 0:
                    continue
                try:
                    cx, cy, _ = trn.circles(f)[0]
                    reward_center_xy = (float(cx), float(cy))
                except Exception:
                    reward_center_xy = None

                for seg in va._iter_between_reward_segment_com(
                    trn,
                    f,
                    fi=fi,
                    df=df,
                    n_buckets=n_buckets,
                    complete=complete,
                    relative_to_reward=True,
                    per_segment_min_meddist_mm=0.0,
                    exclude_wall=False,
                    wc=None,
                    exclude_nonwalk=bool(self.cfg.exclude_nonwalking_frames),
                    nonwalk_mask=nonwalk_mask,
                    min_walk_frames=int(max(2, self.cfg.min_walk_frames or 2)),
                    exclude_reward_endpoints=bool(self.cfg.exclude_reward_endpoints),
                    dist_stats=("max",) if needs_max_frame else (),
                    debug=False,
                    yield_skips=False,
                ):
                    endpoint_offset = 1 if self.cfg.exclude_reward_endpoints else 0
                    s = int(seg.s) + endpoint_offset
                    e = int(seg.e) - endpoint_offset
                    if e <= s:
                        continue
                    if needs_max_frame:
                        max_i = getattr(seg, "max_d_i", None)
                        if max_i is None:
                            continue
                        s_metric = max(s, int(max_i))
                    else:
                        s_metric = s
                    if e <= s_metric:
                        continue

                    min_walk_frames = int(max(2, self.cfg.min_walk_frames or 2))
                    tort = tortuosity_metric_masked(
                        traj=traj,
                        s=s_metric,
                        e=e,
                        fi=fi,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=bool(self.cfg.exclude_nonwalking_frames),
                        px_per_mm=px_per_mm,
                        mode=str(self.cfg.metric_mode or "path_over_max_radius"),
                        reward_center_xy=reward_center_xy,
                        min_keep_frames=min_walk_frames,
                        min_displacement_mm=float(self.cfg.min_displacement_mm or 0.0),
                        min_radius_mm=float(self.cfg.min_radius_mm or 0.0),
                    )
                    if not np.isfinite(tort):
                        continue

                    frac, n_wall, n_frames_eff = self._segment_wall_fraction(
                        traj=traj,
                        s=s_metric,
                        e=e,
                        fi=fi,
                        wc=wc,
                        nonwalk_mask=nonwalk_mask,
                        exclude_nonwalk=bool(self.cfg.exclude_nonwalking_frames),
                        min_walk_frames=min_walk_frames,
                    )
                    if not np.isfinite(frac):
                        continue

                    rows.append(
                        {
                            "wall_frac": float(frac),
                            "wall_pct": float(100.0 * frac),
                            "tortuosity": float(tort),
                            "s": int(seg.s),
                            "e": int(seg.e),
                            "metric_s": int(s_metric),
                            "metric_e": int(e),
                            "n_wall_frames": int(n_wall),
                            "n_metric_frames": int(n_frames_eff),
                            "b_idx": int(getattr(seg, "b_idx", -1)),
                            "video_id": str(video_id),
                            "unit_id": self._unit_id(va, f=f),
                            "fly_id": int(self._fly_id(va, role_idx=role_idx, trx_idx=f)),
                            "trx_idx": int(f),
                            "role_idx": int(role_idx),
                            "fly_role": "exp" if int(role_idx) == 0 else "yok",
                        }
                    )

        keys = [
            "wall_frac",
            "wall_pct",
            "tortuosity",
            "s",
            "e",
            "metric_s",
            "metric_e",
            "n_wall_frames",
            "n_metric_frames",
            "b_idx",
            "video_id",
            "unit_id",
            "fly_id",
            "trx_idx",
            "role_idx",
            "fly_role",
        ]
        out: dict[str, np.ndarray | dict] = {}
        for key in keys:
            vals = [row[key] for row in rows]
            if key in {"video_id", "unit_id", "fly_role"}:
                out[key] = np.asarray(vals, dtype=object)
            elif key in {
                "s",
                "e",
                "metric_s",
                "metric_e",
                "n_wall_frames",
                "n_metric_frames",
                "b_idx",
                "fly_id",
                "trx_idx",
                "role_idx",
            }:
                out[key] = np.asarray(vals, dtype=int)
            else:
                out[key] = np.asarray(vals, dtype=float)

        out["meta"] = {
            "log_tag": self.log_tag,
            "metric": "between_reward_tortuosity_wall_scatter",
            "metric_mode": str(self.cfg.metric_mode),
            "segment_scope": scope,
            "training_index": int(t_idx),
            "training_label": f"training {t_idx + 1}",
            "skip_first_sync_buckets": int(self.cfg.skip_first_sync_buckets),
            "keep_first_sync_buckets": int(self.cfg.keep_first_sync_buckets),
            "exclude_nonwalking_frames": bool(self.cfg.exclude_nonwalking_frames),
            "exclude_reward_endpoints": bool(self.cfg.exclude_reward_endpoints),
            "min_walk_frames": int(self.cfg.min_walk_frames),
            "min_displacement_mm": float(self.cfg.min_displacement_mm),
            "min_radius_mm": float(self.cfg.min_radius_mm),
            "n_segments": int(len(rows)),
            "x_label": "Wall-contact frames in segment [%]",
            "y_label": "Between-reward tortuosity",
        }
        return out

    def export_npz(self, out_npz: str) -> None:
        data = self.collect_records()
        meta = dict(data.pop("meta"))
        util.ensureDir(out_npz)
        np.savez_compressed(out_npz, **data, meta_json=json.dumps(meta, sort_keys=True))
        print(
            f"[{self.log_tag}] wrote {out_npz} "
            f"({int(meta.get('n_segments', 0))} segment(s))"
        )
