import numpy as np
import pytest

from src.analysis.well_contact import detect_well_contacts_edge_or_center


@pytest.mark.parametrize("ref_mode", ["center", "edge"])
def test_large_well_contact_can_classify_interpolated_lost_frames(ref_mode):
    kwargs = {
        "x": np.array([8.0, 0.0, 8.0]),
        "y": np.zeros(3),
        "rot_angles_deg": np.zeros(3),
        "semimaj_ax": np.ones(3),
        "semimin_ax": np.ones(3),
        "lost": np.array([False, True, False]),
        "wells": ((0.0, 0.0),),
        "well_radius": 5.0,
        "ref_mode": ref_mode,
    }

    default_event_contact = detect_well_contacts_edge_or_center(**kwargs)
    event_contact, interpolated_contact = detect_well_contacts_edge_or_center(
        **kwargs, return_interpolated=True
    )

    assert np.isnan(default_event_contact[1])
    assert np.isnan(event_contact[1])
    assert interpolated_contact.tolist() == [0, 1, 0]


def test_large_well_edge_contact_carries_axes_for_interpolated_frame():
    event_contact, interpolated_contact = detect_well_contacts_edge_or_center(
        x=np.array([8.0, 5.5, 8.0]),
        y=np.zeros(3),
        rot_angles_deg=np.zeros(3),
        semimaj_ax=np.array([1.0, np.nan, 1.0]),
        semimin_ax=np.array([1.0, np.nan, 1.0]),
        lost=np.array([False, True, False]),
        wells=((0.0, 0.0),),
        well_radius=5.0,
        ref_mode="edge",
        return_interpolated=True,
    )

    assert np.isnan(event_contact[1])
    assert interpolated_contact.tolist() == [0, 1, 0]


def test_interpolated_hysteresis_does_not_change_event_hysteresis():
    kwargs = {
        "x": np.array([8.0, 0.0, 5.5]),
        "y": np.zeros(3),
        "rot_angles_deg": np.zeros(3),
        "semimaj_ax": np.ones(3),
        "semimin_ax": np.ones(3),
        "lost": np.array([False, True, False]),
        "wells": ((0.0, 0.0),),
        "well_radius": 5.0,
        "ref_mode": "center",
    }

    default_event_contact = detect_well_contacts_edge_or_center(**kwargs)
    event_contact, interpolated_contact = detect_well_contacts_edge_or_center(
        **kwargs, return_interpolated=True
    )

    assert np.allclose(default_event_contact, event_contact, equal_nan=True)
    assert event_contact.tolist()[:1] == [0]
    assert np.isnan(event_contact[1])
    assert event_contact[2] == 0
    assert interpolated_contact.tolist() == [0, 1, 1]
