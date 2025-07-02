

## 🎼 Visual Structure to Sound: Image-Based Graph Sonification Framework

This project is a custom-built **algorithmic composition system** for translating visual input into musical structure. Developed in Python, it operates as a multi-stage pipeline involving **image analysis**, **unsupervised clustering**, **graph-based modeling**, and **MIDI generation**.

Each input image is segmented into clusters according to **shape, contour density, and luminance**. These clusters are used to construct a weighted graph that encodes both **spatial proximity** and **visual intensity**. The resulting graph becomes the blueprint for sound: nodes are mapped to musical parameters such as **pitch, pan, note duration**, and **velocity**, while the structure itself informs **timing** and **gesture**.

The system produces MIDI sequences that are **structurally grounded in the image**, yet musically open—generating forms that are both algorithmically coherent and artistically indeterminate. A dedicated mapping layer ensures **harmonic consistency** (in C major) and introduces additional gestures such as **spatial modulation** and **luminance-derived drones**.

Conceived as an **open compositional instrument**, this system emphasizes not just the sonic output, but the poetics of constraint—the way algorithmic rules shape expressive potential. It is designed for use in **generative sound design**, **visual-music translation**, and **compositional research**.

---



## 🖼️→🎵 Overview

The pipeline follows these main stages:

1. **Image analysis**
   Extract visual attributes (contours, luminance, cluster positions) using `ImageProcessor`.

2. **Graph construction**
   Build a spatial graph from visual data with `GraphBuilder`. Nodes represent clusters; edges are based on proximity.

3. **MIDI generation**
   Translate graph data into music via `SoundMapper`, using node properties (e.g. luminance, position, size) to control pitch, velocity, duration, and spatialization.

---

## ⚙️ Workflow

### 1. Process Images

```python
processor = ImageProcessor(image_path)
processor.load_image()
processor.preprocess_image()
data = processor.compute_attributes()
processor.detect_clusters()
processor.export_json(".../data.json")
processor.draw_contours(".../contour_map.jpg")
```

### 2. Build Graph

```python
builder = GraphBuilder()
builder.add_nodes(data['clusters'], data['dimensions'])
builder.add_edges(max_distance=1000)
builder.compute_weights()
builder.export_graph(".../graph.json")
builder.visualize_graph(".../graph.png")
builder.visualize_graph_on_image(image_path, "graph_overlay.png")
```

### 3. Generate MIDI

```python
mapper = SoundMapper(".../graph.json")
mapper.map_nodes_to_notes("output_sequence.mid")
mapper.create_drone_from_avg_luminance("output_drone.mid")
```

---

## 🎛️ Mapping Logic

* **Pitch** ← average luminance (mapped from 40 to 100, quantized to C major scale)
* **Velocity** ← cluster size
* **Duration** ← scaled with minimum threshold
* **Panning (CC 10)** ← mapped from cluster time or x-position
* **Drone note** ← average luminance across all nodes

```python
pitch = clamp(map_range(luminance, 0, 1, 40, 100))
velocity = clamp(map_range(size, min_size, max_size, 40, 127))
duration = max(MIN_DURATION_TICKS, map_range(size, min_size, max_size, 240, 14400))
```

---

## 📁 File Structure

```
project/
├── src/
│   ├── image_processor.py
│   ├── graph_builder.py
│   └── sound_mapper.py
├── PicturesToAnalysed/
│   ├── images/
│   ├── data/
│   │   ├── json/
│   │   ├── graphs/
│   │   └── midi/
```

---

## 💡 Use Cases

* Generative music from abstract images or photos
* Visual-to-sonic mappings for installation or interactive art
* Sound design based on visual composition
* Embodied score systems for dance or theatre

---

## 🔍 Future Work

* Integrate image features like hue/saturation
* Add real-time MIDI playback
* Incorporate gesture-based control for MIDI mapping
* Visualize MIDI in DAWs or live environments

---

Let me know if you'd like a **French version**, or if you want this as a real `README.md` file with proper headers and links to sample outputs.
