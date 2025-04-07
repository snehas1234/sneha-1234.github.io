import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

image = mpl.image.imread(r"C:\Users\sneha\Pictures\shapes.jpg")
x = image.reshape(-1, 3)

n_clusters = 4
k_means = KMeans(n_clusters=n_clusters, n_init=10)
k_means.fit(x)

segmented_image = k_means.cluster_centers_[k_means.labels_]
segmented_image = segmented_image.reshape(image.shape)

plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.imshow(segmented_image / 255)
plt.title("Segmented Image")
plt.axis("off")
plt.show()

labels_2d = k_means.labels_.reshape(image.shape[:2])

for cluster_id in range(n_clusters):
    extracted = np.zeros_like(image)
    extracted[labels_2d == cluster_id] = image[labels_2d == cluster_id]

    plt.imshow(extracted)
    plt.title(f"Extracted Cluster {cluster_id + 1}")
    plt.axis("off")
    plt.show()
