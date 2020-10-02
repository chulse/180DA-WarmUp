# I used the histogram example and
# https://stackoverflow.com/questions/57608208/whats-the-best-way-to-cut-out-the-center-piece-of-an-image
# to create a bounding box
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)

plt.show()

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    org_h, org_w, _ = img.shape
    new_h, new_w = 100, 100
    crop_img = img[org_h//2 - new_h//2:org_h//2 + new_h, org_w//2 - new_w//2:org_w//2 + new_w,:]
    cv2.imshow("imagec", crop_img)
    crop_img = crop_img.reshape((crop_img.shape[0] * crop_img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(crop_img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    plt.pause(1)

    cv2.rectangle(frame, (org_w//2 - new_w//2, org_h//2 - new_h//2), (org_w//2 + new_w//2, org_h//2 + new_h//2), (0, 255, 0), 2)
    cv2.imshow("image", frame)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

