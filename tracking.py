import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import subprocess

class YOLOTracker:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialise le tracker YOLO
        Args:
            model_path: Chemin vers le modèle YOLO (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            confidence_threshold: Seuil de confiance pour les détections
        """
        print("Chargement du modèle YOLO...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.cap = None
        
        # Couleurs pour les différentes classes
        self.colors = np.random.uniform(0, 255, size=(len(self.model.names), 3))
    
    def find_available_cameras(self):
        """Trouve les caméras disponibles sur le système"""
        print("Recherche des caméras disponibles...")
        available_cameras = []
        
        # Méthode 1: Vérifier les devices video
        try:
            video_devices = subprocess.check_output(['ls', '/dev/video*'], stderr=subprocess.DEVNULL)
            devices = video_devices.decode().strip().split('\n')
            print(f"Devices vidéo détectés: {devices}")
        except subprocess.CalledProcessError:
            print("Aucun device vidéo trouvé dans /dev/video*")
            devices = []
        
        # Méthode 2: Tester les indices de caméra
        for i in range(10):  # Tester les 10 premiers indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        print(f"Caméra trouvée à l'index {i}")
                cap.release()
            except Exception as e:
                pass
        
        return available_cameras
    
    def test_camera_backends(self, camera_index):
        """Teste différents backends OpenCV"""
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_GSTREAMER, "GStreamer"),
            (cv2.CAP_FFMPEG, "FFmpeg"),
            (cv2.CAP_ANY, "Any")
        ]
        
        print(f"Test des backends pour la caméra {camera_index}...")
        
        for backend_id, backend_name in backends:
            try:
                print(f"Test du backend {backend_name}...")
                cap = cv2.VideoCapture(camera_index, backend_id)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ Backend {backend_name} fonctionne!")
                        cap.release()
                        return backend_id
                    else:
                        print(f"✗ Backend {backend_name} - Pas de frame")
                else:
                    print(f"✗ Backend {backend_name} - Impossible d'ouvrir")
                
                cap.release()
            except Exception as e:
                print(f"✗ Backend {backend_name} - Erreur: {e}")
        
        return cv2.CAP_ANY
    
    def start_webcam(self, camera_index=0):
        """Démarre la capture vidéo de la webcam avec diagnostic"""
        print(f"Tentative d'ouverture de la caméra {camera_index}...")
        
        # Vérifier les permissions
        if os.path.exists(f'/dev/video{camera_index}'):
            try:
                permissions = oct(os.stat(f'/dev/video{camera_index}').st_mode)[-3:]
                print(f"Permissions de /dev/video{camera_index}: {permissions}")
            except:
                pass
        
        # Trouver les caméras disponibles
        available_cameras = self.find_available_cameras()
        if not available_cameras:
            print("ERREUR: Aucune caméra disponible détectée!")
            print("Solutions possibles:")
            print("1. Vérifiez que votre webcam est connectée")
            print("2. Exécutez: sudo usermod -a -G video $USER")
            print("3. Redémarrez votre session")
            print("4. Testez avec: v4l2-ctl --list-devices")
            raise ValueError("Aucune caméra disponible")
        
        # Utiliser la première caméra disponible si l'index demandé n'existe pas
        if camera_index not in available_cameras:
            camera_index = available_cameras[0]
            print(f"Utilisation de la caméra {camera_index} à la place")
        
        # Tester les backends
        best_backend = self.test_camera_backends(camera_index)
        
        # Ouvrir la caméra avec le meilleur backend
        self.cap = cv2.VideoCapture(camera_index, best_backend)
        
        if not self.cap.isOpened():
            print("ERREUR: Impossible d'ouvrir la caméra!")
            print("Commandes de diagnostic:")
            print("- lsusb (pour voir les devices USB)")
            print("- v4l2-ctl --list-devices")
            print("- ls -la /dev/video*")
            raise ValueError(f"Impossible d'ouvrir la caméra {camera_index}")
        
        # Test de lecture d'une frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print("ERREUR: Impossible de lire une frame de la caméra!")
            self.cap.release()
            raise ValueError("Caméra ouverte mais pas de données")
        
        print(f"✓ Frame de test lue avec succès: {test_frame.shape}")
        
        # Configuration de la résolution
        print("Configuration de la résolution...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Vérifier la résolution effective
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Webcam initialisée:")
        print(f"  - Résolution: {actual_width}x{actual_height}")
        print(f"  - FPS: {actual_fps}")
        print(f"  - Backend: {best_backend}")
        print(f"  - Index caméra: {camera_index}")
    
    def draw_detections(self, frame, results):
        """Dessine les détections sur l'image"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Coordonnées de la boîte
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Confiance et classe
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Filtrer par seuil de confiance
                    if confidence >= self.confidence_threshold:
                        # Couleur pour cette classe
                        color = self.colors[class_id]
                        
                        # Dessiner la boîte de délimitation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Préparer le texte
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Dessiner le fond du texte
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # Dessiner le texte
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_detection(self):
        """Lance la détection en temps réel"""
        if self.cap is None:
            raise ValueError("Webcam non initialisée. Appelez start_webcam() d'abord.")
        
        print("Détection en cours... Appuyez sur 'q' pour quitter, 's' pour sauvegarder une image")
        
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Erreur lors de la lecture de la frame")
                    break
                
                # Effectuer la détection
                results = self.model(frame, verbose=False)
                
                # Dessiner les détections
                annotated_frame = self.draw_detections(frame.copy(), results)
                
                # Calculer et afficher les FPS
                fps_counter += 1
                if fps_counter % 30 == 0:  # Mise à jour toutes les 30 frames
                    end_time = time.time()
                    fps = 30 / (end_time - start_time)
                    start_time = end_time
                    print(f"FPS: {fps:.1f}")
                
                # Afficher les FPS sur l'image
                cv2.putText(annotated_frame, f"FPS: {fps_counter % 30 * 2:.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Afficher le nombre d'objets détectés
                total_detections = sum(len(result.boxes) if result.boxes is not None else 0 for result in results)
                cv2.putText(annotated_frame, f"Objets detectes: {total_detections}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Afficher l'image
                cv2.imshow('YOLO Detection en Temps Reel', annotated_frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Sauvegarder l'image actuelle
                    timestamp = int(time.time())
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Image sauvegardée: {filename}")
                elif key == ord('c'):
                    # Changer le seuil de confiance
                    self.confidence_threshold = 0.3 if self.confidence_threshold == 0.5 else 0.5
                    print(f"Seuil de confiance changé à: {self.confidence_threshold}")
        
        except KeyboardInterrupt:
            print("\nArrêt demandé par l'utilisateur")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Nettoie les ressources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Ressources nettoyées")

def main():
    """Fonction principale"""
    print("=== Détecteur d'Objets YOLO en Temps Réel ===")
    print("Diagnostic système Ubuntu/Linux...")
    
    # Vérifications préliminaires
    print("\n1. Vérification des permissions...")
    user_groups = subprocess.check_output(['groups']).decode().strip()
    if 'video' in user_groups:
        print("✓ Utilisateur dans le groupe 'video'")
    else:
        print("⚠ Utilisateur PAS dans le groupe 'video'")
        print("Exécutez: sudo usermod -a -G video $USER")
        print("Puis redémarrez votre session")
    
    print("\n2. Vérification des devices vidéo...")
    try:
        devices_output = subprocess.check_output(['ls', '-la', '/dev/video*'], stderr=subprocess.DEVNULL)
        print("Devices vidéo trouvés:")
        print(devices_output.decode())
    except subprocess.CalledProcessError:
        print("Aucun device vidéo trouvé!")
    
    print("\n3. Informations OpenCV...")
    print(f"Version OpenCV: {cv2.__version__}")
    print(f"Backends supportés: {[cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]}")
    
    print("\nControles:")
    print("- 'q': Quitter")
    print("- 's': Sauvegarder l'image actuelle")
    print("- 'c': Changer le seuil de confiance (0.3/0.5)")
    print("=" * 50)
    
    try:
        # Créer le tracker avec un modèle YOLO
        tracker = YOLOTracker(model_path='yolov8n.pt', confidence_threshold=0.5)
        
        # Démarrer la webcam avec diagnostic
        tracker.start_webcam(camera_index=0)
        
        # Lancer la détection
        tracker.run_detection()
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        print("\nSolutions de dépannage Ubuntu:")
        print("1. Vérifiez que votre webcam est connectée et reconnue:")
        print("   lsusb | grep -i camera")
        print("2. Installez les outils v4l2:")
        print("   sudo apt install v4l-utils")
        print("3. Listez les devices vidéo:")
        print("   v4l2-ctl --list-devices")
        print("4. Ajoutez-vous au groupe video:")
        print("   sudo usermod -a -G video $USER")
        print("5. Redémarrez votre session")
        print("6. Testez votre webcam avec:")
        print("   cheese  # ou  vlc v4l2:///dev/video0")

if __name__ == "__main__":
    main()
