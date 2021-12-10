import random

f = open("penguins.csv")
fout = open("penguins_processed.txt", "w")
# Data transformations:
# Labels: [Adelie -> 100, Gentoo -> 010, Chinstrap -> 001]
# Islands: [Torgersen -> 100, Biscoe -> 010, Dream -> 001]
# Bill length: /50 for approx. 0-1 normalization
# Bill depth: /20 for approx. 0-1 normalization
# Flipper length: /200 for approx. 0-1 normalization
# Body mass: /5000 for approx. 0-1 normalization
# Sex: [Male -> 10, Female -> 01]
# Year: [2007 -> 100, 2008 -> 010, 2009 -> 001]

f.readline()  # Destroy header line
lines = []
for line in f:
    if "NA" in line:
        pass
    else:
        linesplit = line.split(",")
        label = {"\"Adelie\"": "1,0,0", "\"Gentoo\"": "0,1,0", "\"Chinstrap\"": "0,0,1"}[linesplit[1]]

        island = {"\"Torgersen\"": "1,0,0", "\"Biscoe\"": "0,1,0", "\"Dream\"": "0,0,1"}[linesplit[2]]
        bill_len = str(float(linesplit[3])/50)
        bill_dep = str(float(linesplit[4])/20)
        flip_len = str(float(linesplit[5])/200)
        body_mas = str(float(linesplit[6])/5000)
        sex = {"\"male\"": "1,0", "\"female\"": "0,1"}[linesplit[7]]
        year = {"2007\n": "1,0,0", "2008\n": "0,1,0", "2009\n": "0,0,1"}[linesplit[8]]
        data = ",".join([island, bill_len, bill_dep, flip_len, body_mas, sex, year])
        data = label + "|" + data

        lines.append(data + "\n")

random.shuffle(lines)
fout.writelines(lines)

f.close()
fout.close()
