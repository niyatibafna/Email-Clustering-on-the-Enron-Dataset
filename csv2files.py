##
 # csv2files.py
 #
 # Archit Checker
 # archit.checker_ug20@ashoka.edu.in
 #
 # Initial processing - read the csv dataset and write contents to text files
 # Every text file - numbered 1 through 517401 has the contents of one email.
 #
 ##

import csv
csv.field_size_limit(256<<15)

i = 0
with open ('dataset/emails/emails.csv', encoding='windows-1252') as file:
    reader = csv.reader(file)
    for row in reader:
        i += 1
        print(i)
        f = open("dataset/emails/processed/"+str(i)+".txt", "w")
        f.write(row[1])
        f.close()
