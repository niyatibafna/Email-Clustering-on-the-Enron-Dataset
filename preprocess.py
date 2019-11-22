import csv
csv.field_size_limit(256<<15)

i = 0
with open ('emails/emails.csv', encoding='windows-1252') as file:
    reader = csv.reader(file)
    for row in reader:
        i += 1
        print(i)
        f = open("emails/processed/"+str(i)+".txt", "w")
        f.write(row[1])
        f.close()

# Leader Clustering