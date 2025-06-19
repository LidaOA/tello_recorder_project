import cv2
from djitellopy import tello
from datetime import datetime
import numpy as np
import os

def record_tello_video_stream(frame_read: tello.BackgroundFrameRead,timestamp):
    """
    Run video recording from Tello drone.
    """
    H,W  = frame_read.frame.shape[:2]  # Dimensions de l'image

    # Crée le fichier vidéo pour l'enregistrement
    out = cv2.VideoWriter(
        f'{timestamp}_tello_recording.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        30,
        (W, H)
    )
    try:
        while True:
            frame = frame_read.frame  # Supposons que c'est une image 2D (grayscale)
            if len(frame.shape) == 2:  # Vérifie si l'image est en niveaux de gris
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convertit en BGR

            frame = frame.astype(np.uint8)
            out.write(frame)
                    
            cv2.imshow("Frame", frame)

            # La touche 'q' permettra de sortir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print('Une erreur est survenue lors de la capture vidéo: ', e)
    finally:
        cv2.destroyAllWindows()
        out.release()

        
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
    