import csv
import sqlite3

from glob import glob
from os.path import expanduser

OUTPUT_FILE = 'dataset.csv'
DATABASE = 'database.sqlite'

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

with open(OUTPUT_FILE, "w", newline='', encoding='UTF-8') as csv_file:  # Python 3 version
    csv_writer = csv.writer(csv_file)
    cursor.execute("select * from Country;")
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(cursor)

    cursor.execute("select * from League;")
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(cursor)

    cursor.execute("select * from Match;")
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(cursor)

    cursor.execute("select * from Player;")
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(cursor)

    cursor.execute("select * from Player_Attributes;")
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(cursor)

    cursor.execute("select * from Team;")
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(cursor)

    cursor.execute("select * from Team_Attributes;")
    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
    csv_writer.writerows(cursor)
