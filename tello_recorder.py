import cv2
from djitellopy import tello
from datetime import datetime
import imageio_ffmpeg
import time
import numpy as np


def record_tello_video_stream(frame_read: tello.BackgroundFrameRead, timestamp):
    """
    Enregistre une vidéo fluide du flux Tello en utilisant imageio_ffmpeg.
    """
    # Obtenir les dimensions du flux vidéo
    original_height, original_width = frame_read.frame.shape[:2]

    # Ajuster les dimensions pour qu'elles soient divisibles par 16
    #height = (original_height + 15) // 16 * 16 
    #width = (original_width + 15) // 16 * 16
    height,width = 720,1280
    size = (width, height)

    # Définir le nom de fichier de sortie
    filename = f"{timestamp}_tello_recording.mp4"

    # Initialiser le writer vidéo
    writer = imageio_ffmpeg.write_frames(filename, size=size, fps=30)
    writer.send(None)

    # Paramètres de framerate
    target_fps = 30
    frame_duration = 1.0 / target_fps
    last_time = time.time()

    prev_frame = None

    try:
        while True:
            # Lire une frame
            frame = frame_read.frame

            # Éviter les frames vides
            if frame is None or frame.size == 0:
                continue

            # Éviter d'écrire la même frame deux fois
            if prev_frame is not None and np.array_equal(frame, prev_frame):
                continue
            prev_frame = frame

            # Redimensionner à la taille souhaitée
            frame = cv2.resize(frame, size)

            # Envoyer au writer
            writer.send(frame)

            # Affichage visuel
            cv2.imshow("Tello Stream", frame)

            # Gérer le framerate
            elapsed = time.time() - last_time
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

            # Sortir si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        writer.close()
        cv2.destroyAllWindows()

        
def main():
    """
    Fonction principale pour la capture vidéo avec Tello.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    drone = tello.Tello()

    try:
        drone.connect()
        print("Connecté au drone Tello")
        print(drone.get_battery())
        drone.streamon()
        print("Stream vidéo activé")
    except Exception as e:
        print("Erreur de connexion ou de démarrage du stream:", e)
        return

    frame_read = drone.get_frame_read()

    # Appelle la fonction de capture
    record_tello_video_stream(frame_read,timestamp)

    print('fin du programme, redémarrage du drone')
    drone.reboot()


if __name__ == "__main__":
    main()
    