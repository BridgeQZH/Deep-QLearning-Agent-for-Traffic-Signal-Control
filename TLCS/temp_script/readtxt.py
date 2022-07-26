from numpy import loadtxt
print("please input which model you are going to test: ")
a = input()
lines = loadtxt("models/model_{}/test/plot_queue_data.txt".format(a), comments="#", delimiter="\n", unpack=False)
newlines = [i for i, e in enumerate(lines) if e != 0]
print(newlines)
print(len(newlines))
print(sum(newlines))
print(sum(newlines)/len(newlines))