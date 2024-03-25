from tqdm import tqdm
from DNNs import GRUNet, OneDimensionalCNN
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from Data_utils import DescriptionSceneDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import train_utilities
import Constants
from argument_parser import argument_parser


def collate_fn(data):
    # desc
    tmp_description = [x[0] for x in data]
    tmp = pad_sequence(tmp_description, batch_first=True)
    descs_ = pack_padded_sequence(tmp,
                                  torch.tensor([len(x) for x in tmp_description]),
                                  batch_first=True,
                                  enforce_sorted=False)
    tmp_scenes = [x[1] for x in data]
    list_length = [len(x[1]) for x in data]
    padded_scenes = pad_sequence(tmp_scenes, batch_first=True)
    padded_scenes = torch.transpose(padded_scenes, 1, 2)
    return descs_, padded_scenes, list_length


def start_train():
    args = argument_parser().parse_args()
    output_feature_size = args.output_feature_size

    is_bidirectional = args.is_bidirectional
    model_descriptor = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=is_bidirectional)
    model_scene = OneDimensionalCNN(in_channels=512, out_channels=512, kernel_size=5, input_size=512,
                                    feature_size=output_feature_size)

    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Loading Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_descriptor.to(device)
    model_scene.to(device)

    # Loading Data
    is_token_level = args.is_token_level
    ret_data = train_utilities.get_entire_data(clip_ones=True, token_level=is_token_level)
    data_description_ = ret_data[Constants.root_description_path]
    data_scene_ = ret_data[Constants.root_scene_path]
    train_indices, val_indices, test_indices = train_utilities.retrieve_indices()
    dataset = DescriptionSceneDataset(data_description_path=data_description_,
                                      data_scene_path=data_scene_,
                                      type_model_scene=Constants.model_scene_onedimensional)

    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False)

    params = list(model_descriptor.parameters()) + list(model_scene.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    train_losses = []
    val_losses = []

    # Define the StepLR scheduler
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    r10_hist = []
    best_r10 = 0

    for _ in tqdm(range(num_epochs)):
        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0

        output_description_val = torch.empty(len(val_indices), output_feature_size)
        output_scene_val = torch.empty(len(val_indices), output_feature_size)

        for i, (data_description, data_scene, length) in enumerate(train_loader):
            data_scene = data_scene.to(device)
            data_description = data_description.to(device)

            optimizer.zero_grad()

            output_descriptor = model_descriptor(data_description)
            output_scene = model_scene(data_scene, length)

            multiplication = train_utilities.cosine_sim(output_descriptor, output_scene)

            loss = train_utilities.contrastive_loss(multiplication)

            loss.backward()

            optimizer.step()

            total_loss_train += loss.item()
            num_batches_train += 1

        scheduler.step()
        print(scheduler.get_last_lr())
        epoch_loss_train = total_loss_train / num_batches_train

        model_descriptor.eval()
        model_scene.eval()

        with torch.no_grad():
            for j, (data_description, data_scene, length) in enumerate(val_loader):
                data_description = data_description.to(device)
                data_scene = data_scene.to(device)
                output_descriptor = model_descriptor(data_description)
                output_scene = model_scene(data_scene, length)

                initial_index = j * batch_size
                final_index = (j + 1) * batch_size
                if final_index > len(val_indices):
                    final_index = len(val_indices)
                output_description_val[initial_index:final_index, :] = output_descriptor
                output_scene_val[initial_index:final_index, :] = output_scene

                multiplication = train_utilities.cosine_sim(output_descriptor, output_scene)

                loss = train_utilities.contrastive_loss(multiplication)

                total_loss_val += loss.item()

                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

        r1, r5, r10, _, _, _, _, _ = train_utilities.evaluate(output_description=output_description_val,
                                                              output_scene=output_scene_val,
                                                              section="val")

        model_descriptor.train()
        model_scene.train()

        r10_hist.append(r10)
        if r10 > best_r10:
            best_r10 = r10
            train_utilities.save_best_model(model_scene.state_dict(), model_descriptor.state_dict(),
                                            'open_clip' + '.pt')

        print("train_loss:", epoch_loss_train)
        print("val_loss:", epoch_loss_val)

        train_losses.append(epoch_loss_train)
        val_losses.append(epoch_loss_val)

    best_model_state_dict_scene, best_model_state_dict_description = train_utilities.load_best_model(
        'open_clip' + '.pt')
    model_scene.load_state_dict(best_model_state_dict_scene)
    model_descriptor.load_state_dict(best_model_state_dict_description)

    model_descriptor.eval()
    model_scene.eval()
    output_description_test = torch.empty(len(test_indices), output_feature_size)
    output_scene_test = torch.empty(len(test_indices), output_feature_size)
    # Evaluate test set
    with torch.no_grad():
        for j, (data_description, data_scene, length) in enumerate(test_loader):
            data_description = data_description.to(device)
            data_scene = data_scene.to(device)
            output_descriptor = model_descriptor(data_description)
            output_scene = model_scene(data_scene, length)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(test_indices):
                final_index = len(test_indices)
            output_description_test[initial_index:final_index, :] = output_descriptor
            output_scene_test[initial_index:final_index, :] = output_scene
    train_utilities.evaluate(
        output_description=output_description_test,
        output_scene=output_scene_test,
        section="test")
    # train_utilities.write_models_evaluation_to_file([ds1, ds5, ds10, sd1, sd5, sd10, ds_medr, sd_medr],
    #                                                 'open_clip_one_dimensional' + "_test_evaluate" + ".txt")
    #
    # train_utilities.plot_procedure(train_losses=train_losses, val_losses=val_losses)


if __name__ == '__main__':
    start_train()
