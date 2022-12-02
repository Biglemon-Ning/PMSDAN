import os
import matplotlib.pyplot as plt

path = os.path.dirname(__file__)
smaple_num = 2817
fig = plt.figure(num = 1, figsize=(5, 5), dpi=200)
plt.xlabel('Training completion rate(%)')
plt.ylabel('Accuracy(%)')
def draw(name):
    with open(path + name, 'r') as f:
        source1 = []
        source2 = []
        source3 = []
        final = []
        maxacc = []
        data = f.read()
        data = data.split()
        for i in range(len(data)):
            if i % 5 == 0:
                source1.append(float(data[i]) / smaple_num * 100)
            elif i % 5 == 1:
                source2.append(float(data[i]) / smaple_num * 100)
            elif i % 5 == 2:
                source3.append(float(data[i]) / smaple_num * 100)
            elif i % 5 == 3:
                final.append(float(data[i]) / smaple_num * 100)
            else:
                maxacc.append(float(data[i]) / smaple_num * 100)
        x = range(len(source1))
        plt.plot(x, source1)
        plt.plot(x, source2)
        plt.plot(x, source3)
        plt.plot(x, final)

if __name__ =='__main__':
    i = 2
    draw('Convergence' + str(i) + '.txt')
    plt.legend(['TAN-WC S1', 'TAN-WC S2', 'TAN-WC S3', 'TAN-WC Final'])
    plt.savefig(r'.\img\Concergence{}'.format(i), dpi = 200)
    plt.show()