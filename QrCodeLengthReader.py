import cv2
from pyzbar.pyzbar import decode
import numpy as np

# Inicializace kamery
cap = cv2.VideoCapture(0)

# Proměnná na sledování, zda má začít skenovat QR kódy
start_scanning = False

# Skutečné rozměry QR kódu (v centimetrech)
qr_code_width_cm = 16.5  # Upravte na skutečné rozměry QR kódu

# Rozlišení obrazu kamery
camera_width = 2688
camera_height = 1520

while True:
    ret, frame = cap.read()

    decoded_objects = decode(frame)

    if start_scanning:
        for obj in decoded_objects:
            # print('Typ: ', obj.type)
            print('Obsah: ', obj.data.decode('utf-8'))

            if obj.type == 'QRCODE':
                points = obj.polygon
                points = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)

                # Získejte šířku QR kódu v pixelech
                qr_width_px = abs(points[0][0] - points[1][0])

                # Vypočítejte vzdálenost od kamery k QR kódu (přibližně)
                distance = (qr_code_width_cm * camera_width) / (2 * qr_width_px)

                print(f'Vzdálenost od kamery k QR kódu: {distance:.2f} cm')

                # Po nalezení prvního QR kódu ukončit skenování
                start_scanning = False

    cv2.imshow("QR Code Scanner", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == ord('s'):
        start_scanning = True

# Uvolnění zdrojů
cap.release()
cv2.destroyAllWindows()
