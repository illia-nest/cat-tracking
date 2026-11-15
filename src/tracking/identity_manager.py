
"""Persistent identity management for Marko and Taras."""

from collections import defaultdict


class PersistentIdentityManager:
    """
    Maps temporary BoT-SORT track IDs to persistent cat identities.

    Identity assignment uses accumulated YOLO class evidence.
    Once assigned, a track's identity cannot change.

    A persistent identity can be released after its track has
    been absent for a configurable number of frames.
    """

    IDENTITIES = ("marko", "taras")

    def __init__(
        self,
        min_observations=12,
        min_class_ratio=0.85,
        owner_timeout_frames=90,
    ):
        self.min_observations = min_observations
        self.min_class_ratio = min_class_ratio
        self.owner_timeout_frames = owner_timeout_frames

        # Accumulated class evidence for each temporary track.
        self.class_scores = defaultdict(
            lambda: defaultdict(float)
        )

        self.observation_counts = defaultdict(int)

        # Permanent assignment for each temporary BoT-SORT ID.
        self.track_identity = {}

        # At most one active track owns each cat identity.
        self.identity_owner = {
            identity: None for identity in self.IDENTITIES
        }

        self.owner_last_seen = {}

    def update(self, result, confirmed_ids, frame_number):
        """
        Update identity evidence for the current frame.

        Returns:
            Dictionary:
                {temporary_track_id: "marko" or "taras"}

            Only currently confirmed and successfully assigned
            tracks are returned.
        """

        boxes = result.boxes

        if boxes is None or boxes.id is None:
            self._release_expired_owners(frame_number)
            return {}

        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        names = result.names

        current_tracks = {}

        # --------------------------------------------------
        # Collect class evidence
        # --------------------------------------------------

        for track_id, class_id, confidence in zip(
            track_ids,
            class_ids,
            confidences,
        ):

            class_name = names[class_id].lower()

            if class_name not in self.IDENTITIES:
                continue

            current_tracks[track_id] = class_name

            # Accumulate confidence-weighted class evidence.
            self.class_scores[track_id][class_name] += confidence
            self.observation_counts[track_id] += 1

        # Release identities whose previous owner disappeared.
        self._release_expired_owners(frame_number)

        # --------------------------------------------------
        # Refresh existing identity assignments
        # --------------------------------------------------

        for track_id in current_tracks:

            identity = self.track_identity.get(track_id)

            if identity is None:
                continue

            owner = self.identity_owner[identity]

            if owner is None or owner == track_id:

                self.identity_owner[identity] = track_id
                self.owner_last_seen[track_id] = frame_number

        # --------------------------------------------------
        # Assign identities to newly confirmed tracks
        # --------------------------------------------------

        for track_id in current_tracks:

            if track_id not in confirmed_ids:
                continue

            # Never change an existing assignment.
            if track_id in self.track_identity:
                continue

            observations = self.observation_counts[track_id]

            if observations < self.min_observations:
                continue

            scores = self.class_scores[track_id]

            total_score = sum(scores.values())

            if total_score <= 0:
                continue

            identity = max(
                self.IDENTITIES,
                key=lambda name: scores[name],
            )

            class_ratio = scores[identity] / total_score

            if class_ratio < self.min_class_ratio:
                continue

            # Don't assign the same identity to two active tracks.
            if self.identity_owner[identity] is not None:
                continue

            self.track_identity[track_id] = identity
            self.identity_owner[identity] = track_id
            self.owner_last_seen[track_id] = frame_number

            print(
                f"[IDENTITY] Track {track_id} assigned to "
                f"{identity.upper()} "
                f"(confidence ratio: {class_ratio:.2f})"
            )

        # --------------------------------------------------
        # Return currently visible confirmed identities
        # --------------------------------------------------

        assignments = {}

        for track_id in confirmed_ids:

            identity = self.track_identity.get(track_id)

            if identity is None:
                continue

            if self.identity_owner[identity] == track_id:
                assignments[track_id] = identity

        return assignments

    def _release_expired_owners(self, frame_number):
        """Release identity ownership after a long absence."""

        for identity, track_id in self.identity_owner.items():

            if track_id is None:
                continue

            last_seen = self.owner_last_seen.get(track_id, frame_number)

            if frame_number - last_seen > self.owner_timeout_frames:

                print(
                    f"[IDENTITY] {identity.upper()} is now absent. "
                    f"Releasing track {track_id}."
                )

                self.identity_owner[identity] = None
                self.owner_last_seen.pop(track_id, None)

    def reset(self):
        """Reset all state before processing another video."""

        self.class_scores.clear()
        self.observation_counts.clear()
        self.track_identity.clear()

        self.identity_owner = {
            identity: None for identity in self.IDENTITIES
        }

        self.owner_last_seen.clear()
