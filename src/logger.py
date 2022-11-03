import csv


class Logger:
    def __init__(self, path):
        self.buffer = list()
        self.path = path
        self.file = open(self.path, "w")
        self.writer = csv.writer(self.file)

    def step(self, reward, done, success):
        self.buffer.append(reward)

        if done:
            self.buffer.insert(0, int(success))
            self.writer.writerow(self.buffer)
            self.file.flush()
            self.buffer.clear()

    def stop(self):
        self.file.close()
        del self.buffer
