"""ANPR validation, HMAC storage and voting (§7.9)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ibvap_worker.anpr import (
    PlateCandidate,
    PlateVoter,
    normalise_plate_text,
    plate_hmac,
    plate_matches,
    to_storage_record,
    validate_indian_plate,
    vote,
)

KEY = b"k" * 32

state = st.sampled_from(["HR", "DL", "MH", "UP", "KA", "TN", "RJ", "PB"])
series = st.text(alphabet="ABCDEFGHJKLMNPQRSTUVWXY", min_size=1, max_size=3)
plates = st.builds(
    lambda s, r, ser, n: f"{s}{r:02d}{ser}{n:04d}",
    state,
    st.integers(1, 99),
    series,
    st.integers(0, 9999),
)


class TestValidation:
    @pytest.mark.parametrize(
        "text",
        ["HR26DA1234", "DL8CAF5030", "MH12AB1234", "22BH1234AA", "KA01A1234"],
    )
    def test_valid_plates(self, text):
        ok, normalised = validate_indian_plate(text)
        assert ok, normalised

    @pytest.mark.parametrize("text", ["", "NOTAPLATE", "12", "HR26DA12E4", "!!!!"])
    def test_invalid_plates(self, text):
        ok, _ = validate_indian_plate(text)
        assert not ok

    def test_separators_and_prefix_stripped(self):
        for variant in [
            "HR 26 DA 1234",
            "HR-26-DA-1234",
            "IND HR26DA1234",
            "hr26da1234",
        ]:
            ok, normalised = validate_indian_plate(variant)
            assert ok and normalised == "HR26DA1234", variant

    def test_confusion_correction_is_position_aware(self):
        """O in a digit slot becomes 0; the same character in a letter slot
        stays O. A global substitution map gets this backwards."""
        ok, normalised = validate_indian_plate("HRZ6DA1Z34")
        assert ok and normalised == "HR26DA1234"

        ok2, norm2 = validate_indian_plate("HR26OA1234")
        assert ok2 and norm2 == "HR26OA1234"  # O is a legitimate series letter

    @given(plate=plates)
    @settings(max_examples=300)
    def test_generated_plates_validate(self, plate):
        ok, normalised = validate_indian_plate(plate)
        assert ok
        assert normalised == plate

    @given(plate=plates)
    @settings(max_examples=300)
    def test_normalisation_is_idempotent(self, plate):
        """Voting compares normalised strings across frames; a non-idempotent
        normaliser would compare apples to pears."""
        once = normalise_plate_text(plate)
        assert normalise_plate_text(once) == once

    @given(text=st.text(max_size=20))
    @settings(max_examples=300)
    def test_never_raises_on_arbitrary_text(self, text):
        ok, normalised = validate_indian_plate(text)
        assert isinstance(ok, bool)
        assert isinstance(normalised, str)

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            validate_indian_plate(None)


class TestPrivacy:
    def test_hmac_is_stable_across_formatting(self):
        assert plate_hmac("HR 26 DA 1234", KEY) == plate_hmac("HR26DA1234", KEY)

    def test_different_keys_give_different_digests(self):
        assert plate_hmac("HR26DA1234", KEY) != plate_hmac("HR26DA1234", b"x" * 32)

    def test_matching_is_constant_time_and_correct(self):
        digest = plate_hmac("HR26DA1234", KEY)
        assert plate_matches("HR26DA1234", digest, KEY)
        assert not plate_matches("MH12AB1234", digest, KEY)

    def test_weak_key_refused(self):
        """A short key makes the digest guessable by enumerating plate space."""
        with pytest.raises(ValueError):
            plate_hmac("HR26DA1234", b"short")
        with pytest.raises(ValueError):
            plate_hmac("HR26DA1234", b"")

    def test_storage_record_holds_no_plaintext_by_default(self):
        """P6: a database leak must not be identifying."""
        read = vote([PlateCandidate("HR26DA1234", 0.9, (0, 0, 1, 1)) for _ in range(3)])
        record = to_storage_record(read, KEY, fired_alert=False)
        assert "plate_text" not in record
        assert record["plate_hmac"] == plate_hmac("HR26DA1234", KEY)

    def test_plaintext_only_inside_a_fired_alert(self):
        read = vote([PlateCandidate("HR26DA1234", 0.9, (0, 0, 1, 1)) for _ in range(3)])
        record = to_storage_record(read, KEY, fired_alert=True)
        assert record["plate_text"] == "HR26DA1234"


class TestVoting:
    def _cands(self, text, n, conf=0.9, char_conf=0.8):
        return [
            PlateCandidate(text, conf, (0, 0, 100, 30), (char_conf,) * len(text), i)
            for i in range(n)
        ]

    def test_requires_agreement(self):
        """Single-frame CCTV OCR is a guess. P3."""
        assert vote(self._cands("HR26DA1234", 1)) is None
        assert vote(self._cands("HR26DA1234", 2)) is None
        assert vote(self._cands("HR26DA1234", 3)) is not None

    def test_winner_is_the_majority(self):
        cands = self._cands("HR26DA1234", 3) + self._cands("MH12AB1234", 1)
        result = vote(cands)
        assert result.text == "HR26DA1234"
        assert result.frames_agreed == 3

    def test_illegible_character_poisons_the_read(self):
        """One character below threshold invalidates the whole plate — a plate
        with one wrong digit is a different vehicle."""
        assert vote(self._cands("HR26DA1234", 5, char_conf=0.2)) is None

    def test_invalid_text_is_never_accepted(self):
        assert vote(self._cands("GARBAGE!!", 10)) is None

    def test_empty_input(self):
        assert vote([]) is None

    def test_bad_threshold_raises(self):
        with pytest.raises(ValueError):
            vote(self._cands("HR26DA1234", 3), min_frames_agreed=0)

    def test_voter_settles_once(self):
        voter = PlateVoter(min_frames_agreed=3)
        results = [voter.add(c) for c in self._cands("HR26DA1234", 5)]
        assert sum(1 for r in results if r is not None) == 1
        assert voter.settled.text == "HR26DA1234"
