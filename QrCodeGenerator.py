import qrcode

# Text, ktorý sa má zakódovať do QR kódu
text = "mam rad ameriku"

# Vytvorte QRCode objekt
qr = qrcode.QRCode(
    version=1,  # Verzia QR kódu (1-40)
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # Úroveň korekcie chýb - L (nízka)
    box_size=10,  # Veľkosť jedného štvorca v QR kode (počet pixelov)
    border=4,  # Hrúbka okraja QR kódu (počet štvorcov)
)

# Pridajte text do QR kódu
qr.add_data(text)
qr.make(fit=True)

# Vytvorte obrázok QR kódu
img = qr.make_image(fill_color="black", back_color="white")

# Uložte obrázok QR kódu do súboru
img.save("qrcode.png")
