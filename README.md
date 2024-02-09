

## Preparing ImageNet64 Dataset with Clustered Label

### Step 1: Acquire ImageNet64

1. Download ImageNet64 from [ImageNet Download Page](https://www.image-net.org/download.php).
2. Extract the `*.zip` files to unveil the dataset batches, including `train_data_batch_1` to `train_data_batch_10` and `val_data`.

Organize the extracted data into:
- Training data: `downloadeddata/train`
- Validation data: `downloadeddata/val`

### Step 2: Setup Environment

Install the required dependencies to ensure the scripts run smoothly:

```bash
pip install -r requirements.txt
```

### Script Execution

- **Extracting Images**: Run `extractimages.py` to decode and store images in PNG format, categorizing them into directories named after ImageNet synset IDs.

- **Data Clustering**: Execute `clustereddata.py` to organize images based on clustered labels. For instance, all cat images (n02124075, n02123394, n02123159, n02123597, n02123045, n02127052) are clustered, enhancing dataset manageability for training purposes. The script also balances the dataset by equalizing the number of images across clusters.

## Additional Resources

For comprehensive label mappings and insights into data clustering, refer to the following resources:
- [ImageNet Full Label Mapping](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57)
- [Clustered Label Mapping](https://github.com/Prev/clustered-imagenet-labels/blob/master/trainer/data/clustered_imagenet_labels.json)
- [Downsampled ImageNet](https://arxiv.org/abs/1707.08819)

