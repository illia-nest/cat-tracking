
from dataclasses import dataclass


@dataclass
class TrackState:
    age: int = 1
    hits: int = 1
    missed: int = 0
    confirmed: bool = False


class TrackConfirmation:
    """
    Filters short-lived BoT-SORT tracks.

    A new track must be observed enough times within a short
    confirmation window before it is considered reliable.
    """

    def __init__(
        self,
        confirmation_window: int = 5,
        required_observations: int = 3,
        min_confidence: float = 0.25,
        max_missed_during_confirmation: int = 2,
        max_missed_after_confirmation: int = 60,
    ):
        self.confirmation_window = confirmation_window
        self.required_observations = required_observations
        self.min_confidence = min_confidence
        self.max_missed_during_confirmation = max_missed_during_confirmation
        self.max_missed_after_confirmation = max_missed_after_confirmation

        self.tracks: dict[int, TrackState] = {}

    def update(self, result) -> set[int]:
        """
        Process one frame's Ultralytics Results object.

        Returns the IDs of confirmed tracks currently detected
        in this frame.
        """

        # Age existing tracks and account for missed observations.
        for state in self.tracks.values():
            state.age += 1
            state.missed += 1

        # Remove expired tentative and confirmed tracks.
        expired = []

        for track_id, state in self.tracks.items():
            if state.confirmed:
                if state.missed > self.max_missed_after_confirmation:
                    expired.append(track_id)
            else:
                if (
                    state.missed > self.max_missed_during_confirmation
                    or state.age > self.confirmation_window
                ):
                    expired.append(track_id)

        for track_id in expired:
            del self.tracks[track_id]

        boxes = result.boxes

        if boxes is None or boxes.id is None:
            return set()

        track_ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        # Process valid observations.
        for track_id, confidence in zip(track_ids, confidences):
            if confidence < self.min_confidence:
                continue

            state = self.tracks.get(track_id)

            if state is None:
                # First observation of a new temporary track.
                self.tracks[track_id] = TrackState()
                continue

            state.hits += 1
            state.missed = 0

            if (
                not state.confirmed
                and state.hits >= self.required_observations
                and state.age <= self.confirmation_window
            ):
                state.confirmed = True

        # Return confirmed tracks observed in this frame.
        confirmed_ids = set()

        for track_id, confidence in zip(track_ids, confidences):
            state = self.tracks.get(track_id)

            if (
                state is not None
                and state.confirmed
                and confidence >= self.min_confidence
            ):
                confirmed_ids.add(track_id)

        return confirmed_ids

    def reset(self):
        """Reset state before processing another video."""
        self.tracks.clear()
