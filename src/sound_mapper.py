import os
import json
import math
from mido import MidiFile, MidiTrack, Message, MetaMessage


def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def clamp(value, min_value=0, max_value=127):
    return max(min_value, min(value, max_value))


def enforce_c_major_scale(pitch):
    # Define C major scale in MIDI pitches
    c_major_scale = [0, 2, 4, 5, 7, 9, 11]
    octave = pitch // 12
    note_in_octave = pitch % 12

    # Find the closest note in the C major scale
    closest_note = min(c_major_scale, key=lambda n: abs(n - note_in_octave))
    return octave * 12 + closest_note


class SoundMapper:
    def __init__(self, graph_path):
        self.graph_path = graph_path
        self.graph_data = self.load_graph()

    def load_graph(self):
        try:
            with open(self.graph_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading graph {self.graph_path}: {e}")
            raise

    def map_nodes_to_notes(self, output_midi_path):
        nodes = self.graph_data.get('clusters', [])
        if not nodes:
            print(f"No clusters found in {self.graph_path}")
            return

        # Sort nodes by time for sequential order
        nodes = sorted(nodes, key=lambda n: n.get('time', 0))

        total_time = max(node.get('time', 0) for node in nodes) if nodes else 1
        min_size = min(node.get('size', 0) for node in nodes)
        max_size = max(node.get('size', 0) for node in nodes)

        # Debugging: Check the range of sizes
        print(f"Min size: {min_size}, Max size: {max_size}")

        # Define minimum note duration in ticks
        MIN_DURATION_TICKS = 4800  # 5 seconds

        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)

        base_name = os.path.basename(self.graph_path)
        image_name = base_name.split("_graph_")[0]
        track.append(MetaMessage('track_name', name=image_name))

        current_time = 0
        for node in nodes:
            try:
                luminance = node['avg_luminance'] / 255
                pitch = clamp(int(map_range(luminance, 0, 1, 40, 100)))
                # Adjust pitch to C major scale
                pitch = enforce_c_major_scale(pitch)
                node_time = node.get('time', 0)

                # Map time to pan position (0-127)
                pan = clamp(int(map_range(node_time, 0, total_time, 0, 127)))
                velocity = clamp(
                    int(map_range(node['size'], min_size, max_size, 40, 127)))

                # Simplified mapping of size to duration with minimum duration
                raw_duration = int(
                    map_range(node['size'], min_size, max_size, 240, 14400))
                duration = max(raw_duration, MIN_DURATION_TICKS)

                # Debugging: Check values for each cluster
                print(
                    f"Cluster: {node}, Pitch: {pitch}, Pan: {pan}, Size: {node['size']}, Raw Duration: {raw_duration}, Final Duration: {duration}")

                delta_time = int(node_time * 480) - current_time
                current_time = int(node_time * 480)

                track.append(Message('note_on', note=pitch,
                             velocity=velocity, time=max(delta_time, 0)))
                track.append(Message('control_change',
                             control=10, value=pan, time=0))
                track.append(Message('note_off', note=pitch,
                             velocity=velocity, time=duration))

            except Exception as e:
                print(f"Error processing node {node}: {e}")

        midi.save(output_midi_path)
        if os.path.exists(output_midi_path):
            print(f"MIDI file successfully created: {output_midi_path}")
        else:
            print(f"Failed to save MIDI file: {output_midi_path}")

    def create_drone_from_avg_luminance(self, output_midi_path):
        # Calculate average luminance of all clusters
        nodes = self.graph_data.get('clusters', [])
        if not nodes:
            print(f"No clusters found in {self.graph_path}")
            return

        avg_luminance = sum(node['avg_luminance']
                            for node in nodes) / len(nodes)
        normalized_luminance = avg_luminance / 255

        # Map luminance to a MIDI pitch (40-100)
        pitch = clamp(int(map_range(normalized_luminance, 0, 1, 40, 100)))
        pitch = enforce_c_major_scale(pitch)  # Adjust pitch to C major scale

        # Create MIDI file
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)

        base_name = os.path.basename(self.graph_path)
        image_name = base_name.split("_graph_")[0]
        track.append(MetaMessage('track_name', name=f"Drone_{image_name}"))

        # Add a sustained note for the drone
        velocity = 100  # Fixed velocity for the drone
        duration = 4800  # Duration of the note (long sustained)
        track.append(Message('note_on', note=pitch, velocity=velocity, time=0))
        track.append(Message('note_off', note=pitch,
                     velocity=velocity, time=duration))

        # Save the MIDI file
        midi.save(output_midi_path)
        if os.path.exists(output_midi_path):
            print(f"Drone MIDI file successfully created: {output_midi_path}")
        else:
            print(f"Failed to save Drone MIDI file: {output_midi_path}")
