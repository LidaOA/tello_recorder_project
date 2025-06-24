import cv2
import imageio_ffmpeg

def convert_bgr_to_rgb_with_imageio(video_path, output_path):
    # Ouvrir la vidéo source avec OpenCV
    cap = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo est ouverte
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Obtenir les dimensions et le framerate de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (width, height)

    # Initialiser le writer avec imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(output_path, size=size, fps=fps)
    writer.send(None)  # Initialisation du flux

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Envoyer la frame au writer
        writer.send(frame_rgb)

    # Libérer les ressources
    cap.release()
    writer.close()

# Exemple d'utilisation
input_video_path = "20250622_171827_tello_recording.mp4"
output_video_path = "20250622_171827_tello_recording_rgb.mp4"
convert_bgr_to_rgb_with_imageio(input_video_path, output_video_path)