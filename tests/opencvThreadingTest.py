import queue
import threading
import time
import cv2
import numpy as np

exitFlag = 0
queueLock = threading.Lock()
grayQueue = queue.Queue(100)
edgeQueue = queue.Queue(100)
edges = np.zeros((480, 640, 1))
gray = np.zeros((480, 640, 1))


class myThread (threading.Thread):
    def __init__(self, threadID, name, function, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.function = function
        self.q = q

    def run(self):
        print("Starting " + self.name)
        self.function(self.name, self.q)
        print("Exiting " + self.name)


def bgr2gray(threadName, q):
    global gray
    while not exitFlag:
        if not q.empty():
            data = q.get()
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            edgeQueue.put(gray)


def edge(threadName, q):
    global edges
    while not exitFlag:
        if not q.empty():
            data = q.get()
            edges = cv2.Canny(data, 100, 200)


def main():
    threadList = ["bgr2gray", "edge detector"]
    threads = []
    threadID = 0
    global exitFlag

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Create new threads
    thread = myThread(threadID, threadList[threadID], bgr2gray, grayQueue)
    thread.start()
    threads.append(thread)
    threadID += 1
    thread = myThread(threadID, threadList[threadID], edge, edgeQueue)
    thread.start()
    threads.append(thread)

    # Wait for queue to empty
    while True:
        queueLock.acquire()
        ts = time.time()
        _, frame = cap.read()
        grayQueue.put(frame)
        queueLock.release()
        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        cv2.imshow("edges", edges)
        te = time.time()
        print(te-ts)
        if cv2.waitKey(1) == 27:
            break

    # Notify threads it's time to exit
    exitFlag = 1

    # Wait for all threads to complete
    for t in threads:
        t.join()
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting Main Thread")


if __name__ == "__main__":
    main()
