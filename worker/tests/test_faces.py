"""Face matching against a curated watchlist (§7.10). OPT-IN.

Naming a specific human being is the highest-stakes false positive in this
system (P3, P6), so the tests here weight toward what must NOT match, and
toward the privacy invariant: nothing here ever writes an embedding anywhere
except the caller-supplied watchlist it was given to check against.
"""

from __future__ import annotations

import pytest

from ibvap_worker.faces import (
    FaceConfig,
    FaceMatch,
    FaceVoter,
    FaceWatchlist,
    WatchlistFace,
    cosine_similarity,
    vote_face,
)


def unit(*values: float) -> tuple[float, ...]:
    """A vector, not normalised -- cosine_similarity must normalise itself."""
    return tuple(values)


class TestCosineSimilarity:
    def test_identical_vectors_are_1(self):
        v = unit(1.0, 2.0, 3.0)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors_are_minus_1(self):
        assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)

    def test_orthogonal_vectors_are_0(self):
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_scale_invariant(self):
        """An embedder that returns unnormalised vectors must not silently
        produce similarities above 1 -- this is what makes that safe."""
        a = unit(1.0, 2.0, 3.0)
        b = tuple(x * 100.0 for x in a)
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_dimension_mismatch_raises(self):
        """Comparing a 512-d embedding with a 128-d one is a configuration
        bug, not a runtime condition to paper over with a plausible number."""
        with pytest.raises(ValueError):
            cosine_similarity((1.0, 2.0), (1.0, 2.0, 3.0))

    def test_zero_vector_is_zero_not_a_division_error(self):
        assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0


class TestFaceWatchlist:
    def _wl(self, threshold=0.55):
        wl = FaceWatchlist(threshold=threshold)
        wl.set_faces(
            [
                WatchlistFace("p1", "REF-1", "watch", unit(1.0, 0.0, 0.0)),
                WatchlistFace("p2", "REF-2", "watch", unit(0.0, 1.0, 0.0)),
            ]
        )
        return wl

    def test_exact_match(self):
        got = self._wl().match(unit(1.0, 0.0, 0.0))
        assert got is not None and got.person_id == "p1"

    def test_below_threshold_is_no_match(self):
        assert self._wl().match(unit(0.5, 0.5, 0.7)) is None

    def test_returns_only_the_single_best(self):
        """Handing an operator a ranked list invites them to pick the answer
        that fits their theory. One answer or none."""
        wl = FaceWatchlist(threshold=0.1)
        wl.set_faces(
            [
                WatchlistFace("weak", "REF-W", "watch", unit(0.9, 0.1, 0.0)),
                WatchlistFace("strong", "REF-S", "watch", unit(1.0, 0.0, 0.0)),
            ]
        )
        got = wl.match(unit(1.0, 0.0, 0.0))
        assert got is not None and got.person_id == "strong"

    def test_empty_watchlist_matches_nobody(self):
        assert FaceWatchlist(threshold=0.5).match(unit(1.0, 0.0)) is None

    def test_count(self):
        assert self._wl().count == 2


class TestVoteFace:
    def test_a_single_frame_does_not_name_anyone(self):
        m = FaceMatch("p1", "REF-1", "watch", 0.9)
        assert vote_face([m], min_frames_agreed=2) is None

    def test_settles_once_enough_frames_agree_on_the_same_person(self):
        m = FaceMatch("p1", "REF-1", "watch", 0.9)
        got = vote_face([m, m], min_frames_agreed=2)
        assert got is not None and got.person_id == "p1" and got.frames_agreed == 2

    def test_two_different_people_never_average_into_a_false_confident_answer(self):
        """The case that justifies requiring plurality on WHO, not just on
        'matched somebody' the way weapon.py votes on 'armed'. Two people
        drawing similar scores must not blend into a match for either."""
        a = FaceMatch("p1", "REF-1", "watch", 0.9)
        b = FaceMatch("p2", "REF-2", "watch", 0.9)
        assert vote_face([a, b, a, b], min_frames_agreed=3) is None

    def test_quiet_frames_do_not_count(self):
        m = FaceMatch("p1", "REF-1", "watch", 0.9)
        assert vote_face([m, None, m, None], min_frames_agreed=3) is None

    def test_rejects_nonsense_threshold(self):
        with pytest.raises(ValueError):
            vote_face([], min_frames_agreed=0)


class TestFaceVoter:
    def test_a_held_match_settles(self):
        voter = FaceVoter(min_frames_agreed=2, window_frames=6)
        m = FaceMatch("p1", "REF-1", "watch", 0.9)
        assert voter.add(m) is None
        got = voter.add(m)
        assert got is not None and got.person_id == "p1"

    def test_turning_away_un_settles(self):
        voter = FaceVoter(min_frames_agreed=2, window_frames=3)
        m = FaceMatch("p1", "REF-1", "watch", 0.9)
        voter.add(m)
        voter.add(m)
        last = None
        for _ in range(3):
            last = voter.add(None)
        assert last is None

    def test_reset(self):
        voter = FaceVoter(min_frames_agreed=1, window_frames=6)
        voter.add(FaceMatch("p1", "REF-1", "watch", 0.9))
        voter.reset()
        assert voter.add(None) is None


class TestFaceConfig:
    def test_default_is_disabled(self):
        """P6: the answer to the privacy question is a default, not a
        suggestion."""
        assert FaceConfig().enabled is False

    def test_default_never_retains_non_matching_embeddings(self):
        assert FaceConfig().retain_non_matching_embeddings is False

    def test_from_mapping_reads_the_block(self):
        cfg = FaceConfig.from_mapping(
            {
                "faces": {
                    "enabled": True,
                    "match_threshold": 0.7,
                    "min_face_px": 60,
                    "voting": {"min_frames_agreed": 3, "window_frames": 10},
                }
            }
        )
        assert cfg.enabled and cfg.match_threshold == 0.7 and cfg.min_face_px == 60
        assert cfg.min_frames_agreed == 3 and cfg.window_frames == 10

    def test_every_n_frames_is_never_zero(self):
        assert FaceConfig.from_mapping({"faces": {"every_n_frames": 0}}).every_n_frames == 1
