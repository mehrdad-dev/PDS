import torch
from torch import nn
from tqdm import tqdm
from glob import glob
import os
from torchvision.utils import save_image
from PIL import Image
import random
from torch.utils.data import DataLoader


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


def expand_dataset(train_path, transform):
    for clas in tqdm(glob(os.path.join(train_path, '*'))):
        class_name = clas.split('/')[-1]
        images_path = glob(os.path.join(clas, '*'))
        for expand in range(2):
            counter = 1
            for img in glob(os.path.join(clas, '*')):
                image = Image.open(img)
                image = transform(image)
                randn = random.randrange(0,1000)
                save_image(image, clas+f'/{class_name}-{counter}-{randn}.png')
                counter +=1


def resize_dataset(path, transform):
    for clas in glob(os.path.join(path, '*')):
        class_name = clas.split('/')[-1]
        for img in glob(os.path.join(clas, '*')):
            image = Image.open(img)
            image = transform(image)
            save_image(image, img)


def file2classid(path):
    file2classid_list = []
    for clas in glob(os.path.join(path, '*')):
        class_name = clas.split('/')[-1]
        class_id = 0 if class_name == 'parkinson' else 1
        for img in glob(os.path.join(clas, '*')):
            img_name = os.path.basename(img)
            file2classid_list.append((img_name, class_id))
    return file2classid_list


def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
            .detach()
            .data,
            1,
        )[1]
        == query_labels.cuda()
    ).sum().item(), len(query_labels)


def evaluate(model, data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(data_loader), total=len(data_loader)) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                correct, total = evaluate_on_one_task(model, support_images,
                                    support_labels, query_images, query_labels)

                total_predictions += total
                correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )


def fit(
    model,
    optimizer,
    criterion,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()

    return loss.item()

    