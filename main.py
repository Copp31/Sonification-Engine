from src.image_processor import ImageProcessor
from src.graph_builder import GraphBuilder
from src.sound_mapper import SoundMapper
import os
import glob
from datetime import datetime
import shutil

print(
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ฅ^•ﻌ•^ฅ WELCOME TO THE IMAGE SONIFICATION TOOL! ฅ^•ﻌ•^ฅ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹ Exploring images and converting them into soundscapes ⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫⊰⊹⊹⊱✫\n"
    "｡ﾟ•┈୨♡୧┈•ﾟ｡｡ﾟ•┈୨♡୧┈•ﾟ｡｡ﾟ•┈୨♡୧┈•ﾟ｡ Processing visual textures, patterns, and moments... ｡ﾟ•┈୨♡୧┈•ﾟ｡｡ﾟ•┈୨♡୧┈•ﾟ｡｡ﾟ•┈୨♡୧┈•ﾟ｡｡ﾟ•┈୨♡୧┈•ﾟ｡｡ﾟ\n"
    "°•. ✿ .•°•. ✿ .•°°•. ✿ .•°•. ✿ .•° Scanning... Decoding... Weaving visuals into sound °•. ✿ .•°•. ✿ .•°°•. ✿ .•°•. ✿ .•°°•. ✿ .\n\n"
    "⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸ Let the harmony of pixels and sounds guide you ⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸⫷⫸\n\n"
    "⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉ Sit back, relax, and let the data play its tune ⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉⧉\n\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Let's make something beautiful! ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░"
)


def prepare_folders(*folders):
    """
    Prepare folders by ensuring they exist and are empty.

    Args:
        *folders (str): List of folder paths to prepare.
    """
    for folder in folders:
        if os.path.exists(folder):
            # Supprimer tout le contenu du dossier
            shutil.rmtree(folder)
        # Recréer le dossier
        os.makedirs(folder)



# Base path : remonte depuis "script/" vers "CODING/"
base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PicturesToAnalysed"))
image_folder = os.path.join(base_folder, "images")
output_json_folder = os.path.join(base_folder, "data", "json")
output_contour_folder = os.path.join(image_folder, "contourMaps")
output_graph_folder = os.path.join(base_folder, "data", "graphs")

folders = [image_folder, output_json_folder, output_contour_folder, output_graph_folder]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Dossier créé : {folder}")
    else:
        print(f"Dossier existant : {folder}")

for folder in folders:
    print(f"Chemin absolu : {os.path.abspath(folder)}")

prepare_folders(output_json_folder, output_contour_folder, output_graph_folder)

for folder in folders:
    if not os.path.exists(folder):
        print(f"Dossier manquant : {folder}")
        
image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
for image_path in image_files:
    image_name = os.path.basename(image_path)


    if not os.path.exists(image_path):
        print(f"Image {image_name} not found, skipping.")
        continue

    print(f"Processing {image_name}...")

    # Générer un horodatage pour chaque fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Étape 1 : Analyse de l'image
    processor = ImageProcessor(image_path)
    processor.load_image()
    processor.preprocess_image()
    data = processor.compute_attributes()
    processor.detect_clusters()
    processor.export_json(
        os.path.join(output_json_folder, f"{os.path.splitext(image_name)[0]}_data_{timestamp}.json"))
    processor.draw_contours(
        output_path=os.path.join(output_contour_folder, f"{os.path.splitext(image_name)[0]}_contours_{timestamp}.jpg"))
    processor.draw_contours_on_black(
        output_path=os.path.join(output_contour_folder, f"{os.path.splitext(image_name)[0]}_contoursBLACK_{timestamp}.png"))

    # Étape 2 : Construction du graphe
    builder = GraphBuilder()
    builder.add_nodes(data['clusters'], data['dimensions'])
    # Relie les nœuds à une distance <= 150
    builder.add_edges(max_distance=1000)
    builder.compute_weights()
    builder.export_graph(
        os.path.join(output_graph_folder, f"{os.path.splitext(image_name)[0]}_graph_{timestamp}.json"))
    builder.visualize_graph(
        os.path.join(output_graph_folder, f"{os.path.splitext(image_name)[0]}_graph_{timestamp}.png"))
    builder.visualize_graph_on_image(
        image_path, os.path.join(output_graph_folder, f"{os.path.splitext(image_name)[0]}_graph_overlay_{timestamp}.png"))
    builder.visualize_graph_like_overlay_but_without_image(
        image_path=image_path,
        output_path=os.path.join(output_graph_folder, f"{os.path.splitext(image_name)[0]}_graph_overlay_noimage_{timestamp}.png"))

    print("Données extraites :")
    print(data)
    print(f"Finished processing {image_name}")


# Étape 3 : Mapping sonore
if __name__ == "__main__":
    # Définition du dossier de base (remonte depuis le script)
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PicturesToAnalysed"))

    # Définition des chemins avec os.path.join() (compatibles Mac & Windows)
    graph_folder = os.path.join(base_folder, "data", "graphs")
    json_output_folder = os.path.join(base_folder, "data", "json")
    midi_output_folder = os.path.join(base_folder, "data", "midi")

    folders = [graph_folder, json_output_folder, midi_output_folder]

    # Vérifier et créer les dossiers si manquants
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"Dossier créé : {folder}")
        else:
            print(f"Dossier existant : {folder}")

    # Afficher les chemins absolus pour vérification
    print("\n📂 Chemins absolus des dossiers :")
    for folder in folders:
        print(f"  - {os.path.abspath(folder)}")

    # Parcourir tous les fichiers JSON dans le dossier des graphes
    graph_files = glob.glob(os.path.join(json_output_folder, "*.json"))

    if not graph_files:
        print("\n⚠️ Aucun fichier JSON trouvé dans:", json_output_folder)

    for graph_path in graph_files:
        # Extraire le nom de l'image à partir du fichier JSON
        # Exemple : 05_data_20240607_143300.json
        base_name = os.path.basename(graph_path)
        image_name = base_name.split("_data_")[0]  # Exemple : SEENE_Sarah_05

        # Nom du fichier MIDI basé sur l'image
        output_midi_path = os.path.join(midi_output_folder, f"output_{image_name}.mid")
        output_midi_path_DRONE = os.path.join(midi_output_folder, f"output_{image_name}_DRONE.mid")

        print(f"\n Processing graph: {graph_path}")
        print(f" Output MIDI: {output_midi_path}")

        try:
            mapper = SoundMapper(graph_path)

            # Vérification des méthodes avant exécution
            if hasattr(mapper, "map_nodes_to_notes"):
                mapper.map_nodes_to_notes(output_midi_path)
            else:
                print(f"⚠️ Warning: La méthode 'map_nodes_to_notes' est introuvable dans SoundMapper.")

            if hasattr(mapper, "create_drone_from_avg_luminance"):
                mapper.create_drone_from_avg_luminance(output_midi_path_DRONE)
            else:
                print(f"⚠️ Warning: La méthode 'create_drone_from_avg_luminance' est introuvable dans SoundMapper.")

        except Exception as e:
            print(f" Erreur lors du traitement de {graph_path}: {e}")

    print("\n ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░Finished processing all graph files. ฅ^•ﻌ•^ฅ CONGRATULATIONS! ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
    
# # Étape 4 : Génération audio
# generator = AudioGenerator("data/midi/output.mid")
# generator.add_effects()
# generator.export_audio("data/audio/output.wav")
