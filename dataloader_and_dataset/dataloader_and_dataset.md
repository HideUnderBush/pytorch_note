# Pytorch101: Dataset and Dataloader
It was quite confusing for me to use the ``DataLoader`` when I built my first Pytorch project, maybe just like some of you guys, I copied and pasted the code from the official tutorial and found that image data are loaded successfully with correctly allocated batches. However, it did take me longer time to understand the inner mechanism of the ``Dataset`` and ``DataLoader`` classes.

## What kind of data we need to load
It is important to determine what kinds of data that we need to load for our project since it decides the design of our ``__getitem__`` function as well as how to manage data in disks. Currently, the package ``torchvision`` provides data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. To load those data, it simply needs only one-line code: ``trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)`` where you may need to define the ``transform`` by yourself. For general classification problems, we may also need labels/classes of corresponding data. Moreover, with more complex projects, other information may be used as input, for example, we may use landmarkers of human face within a face classification model. In this case, we need to load face images, labels and landmarkers.

## How to load data
Basically, we use ``Dataset`` class to load datasets from disks. The ``Dataset`` class is actually an abstract class, and in most cases, we need to override its functions by creating a child class. If you are familiar with C++ virtual function then this is easier to understand. It is like, the ``Dataset`` class contains virtual functions like ``__init__``, ``__getitem__`` without specific definitions. If we want to load our own data, we need to defined those functions in an inherited class. So, to load our own data (especially that we want to load some auxiliary data in addition to input images), we need to (1) create a child class ``class SelfDefinedDataset(Dataset): ``, and then (2) override the related functions like ``__init__``, ``__getitem__``, ``__len()`` and etc.
the original ``__getitem__``function takes an index as the input. the general way to load our own data, is to
* write a function ``my_fn_collect_data`` collecting all the data with its related attributes.
* make sure the return value of ``my_fn_collect_data`` is a list-like item, data entries (including images, labels and other information) can be access by an index.  
(how to build up connection between the data path and ``Dataset``, how to arrange the data in disks(e.g. use folders? different name schema? how to link the label with the data))
* override ``__getitem__`` by calling the ``my_fn_collect_data`` to obtain data entries with specific index.  

Of course there are other ways to manage your data, the proposed steps only show a general way of doing so. In this post, we will present an RMB classification example that requires image data and related labels. Assuming that we have anotated images of two types of RMB (1 yuan and 100 yuan), and they are placed in two folders "1" and "100" respectively.
The data directory is constructed in the following way:
```
data_dir/
  - 1/
    -1.jpg
    -2.jpg
    ...
  - 100/
    -1.jpg
    -2.jpg
    ...
```

Let's first have a look at the source code of ``Dataset`` class:
```python
class Dataset(object):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py
```
As mentioned before, we need to override this class to build our own "Dataset" class. Before that, we need to write a function to "collect" data from disk:
```Python
rmb_label = {"1":0, "100":1}
class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  
        # all the images and related labels are stored in data_info
        # and it can be access by index (list-like)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   
            # image pre-process, will be discussed later
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # traverse labels
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # traverse images
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    # rmb_label is a dict defined at the beginning
                    data_info.append((path_img, int(label)))

        return data_info
```
Note that the function ``get_img_info`` traverses the data folder and collects all the images with their labels, and the return value ``data_info`` is a list that items inside can be accessed by index. You may also find that using ``tuple`` to store one data entry is a pretty good idea as items of tuple can be different types (images, integral and even list), and the return value can therefore be a list of tuples.
With this function, we can then rewrite the ``__getitem__`` function fetching single data entry ``path_img, label = self.data_info[index]`` to access the images and labels. We also add ``transform`` here for data pre-processing, which will be detailed later. In addition, we can also rewrite the ``__len__`` function to return the length of our dataset, that is the total number of our data entries. Till here, we almost done the loading data part, however, we still need to process the data to make them fit our training propuse.
## Data preprocessing  
You may notice before that we used a ``transform`` function in the ``RMBDataset`` and it is also the input of the ``__init__`` function. Generally, we precess our data through this function. Transform are commonly used image transformation methods that can be chained together using ``compose``, here is an example:
```
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```
Images will then be processed by a series of transformation listed in ``Compose()``. Commonly used transformation includes ``Resize()``, ``RandomCrop``, ``CenterCrop()``, ``ToTensor()``, ``Normalize()`` and ``Grayscale()``, more options and detailed usage can be found [here](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms)
After creating a series of transformation, we pass it to our defined dataset as input. ``train_data = RMBDataset(data_dir=train_dir, transform=train_transform)`` to make sure the training data be processed before training.
## Allocate data in mini batches
allocate with default functions, drop last, shuffle, load data in parallel.
