
# Using readlines()
file1 = open('data.txt', 'r')
Lines = file1.readlines()

count = 1
for line in Lines:
    line = line.replace(',','.').strip()
    f = open("seed_"+str(count)+".csv", "a")
    f.write("num_evaluations,best_reward,best_program\n")
    f.write("1,"+str(line)+',')
    f.close()
    count+=1
