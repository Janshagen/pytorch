"""Flytta raw bilder som inte sparats som jpeg på onedrive till mappen raderade"""
import os

# Input av filsökväg
year_found = False
year = input("Ange år: ")
while not year_found:
    try:
        os.listdir("C:/Bild&Film/" + year)
        year_found = True
    except FileNotFoundError:
        year = input(year + " har inte varit än, ange år igen: ")

subject_found = False
subject = input("Ange ämne: ")
while not subject_found:
    try:
        os.listdir("C:/Bild&Film/" + year + "/" + subject)
        subject_found = True
    except FileNotFoundError:
        subject = input("Du har inte gjort " + subject + ", prova igen: ")

folder_found = False
folder = input("Ange folder, (om ingen undermapp, skriv 'alla'): ")
if folder == 'alla':
    while not folder_found:
        try:
            arw = os.listdir("C:/Bild&Film/" + year + "/" + subject)
            jpeg = os.listdir("C:/Users/anton/OneDrive/Pictures/Lightroom Saved Photos/"
                              + year + "/" + subject)
            folder_found = True
        except FileNotFoundError:
            folder = input("Foldern hittades inte, ange ny folder: ")
else:
    while not folder_found:
        try:
            arw = os.listdir("C:/Bild&Film/" + year + "/" + subject + "/" + folder)
            jpeg = os.listdir("C:/Users/anton/OneDrive/Pictures/Lightroom Saved Photos/"
                              + year + "/" + subject + "/" + folder)
            folder_found = True
        except FileNotFoundError:
            folder = input("Foldern hittades inte, ange ny folder: ")

# Döper om jpeg till ARW för att kunna jämföra
ny_jpeg = []
for file in jpeg:
    cut = file.replace('.jpg', '.ARW')
    ny_jpeg.append(cut)

# Flyttar arw-filerna till mappen "raderade"
deleted = []
if folder == 'alla':
    for raw in arw:
        if not ny_jpeg.count(raw):
            os.replace("C:/Bild&Film/" + year + "/" + subject + "/" + raw, "C:/Bild&Film/Raderade/" + raw)
            deleted.append(raw)
else:
    for raw in arw:
        if not ny_jpeg.count(raw):
            os.replace("C:/Bild&Film/" + year + "/" + subject + "/" + folder + "/" + raw, "C:/Bild&Film/Raderade/" + raw)
            deleted.append(raw)

print(deleted)
print('totalt', len(deleted), 'raderade filer')
