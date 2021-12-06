import torch
from torch.autograd import Variable
from torchvision import transforms

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from datasets import VehicleDataset

def test(net, batch_size):

    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std = [0.5, 0.5, 0.5])
    ])

    testset = VehicleDataset('./data/val', transform=test_transform, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4, shuffle=False)

    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda")
    predictions = []

    for batch_idx, inputs in tqdm(enumerate(testloader)):
        with torch.no_grad():
            idx = batch_idx
            inputs  = inputs.to(device)
            # inputs = Variable(inputs, volatile=True)
            output_1, output_2, output_3, output_concat= net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            
            # preds = predicted.argmax(-1).to('cpu').tolist()
            preds = predicted.to('cpu').tolist()
            
            # # preds <- argmax output
            # # predictions extend pred
            predictions.extend(preds)

    # return test_acc, test_acc_en, test_loss
    submission(testloader.dataset.images, predictions)


def encoding_name(filename):
    return os.path.basename(filename).split('.')[0]

def submission(file_name, preds, path='./results', team_name='aiteam'):

    save_path = os.path.join(path, 'output')
    os.makedirs(save_path, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = os.path.join(save_path, f'{team_name}_{today}.csv')

    data = np.stack([np.array(file_name), np.array(preds)], axis=1)

    submission = pd.DataFrame(data=data, columns=['encoded_name', 'label'])
    submission['encoded_name'] = submission['encoded_name'].apply(encoding_name)
    submission.to_csv(csv_name,index=None)
    
    
    print(f'|INFO| DATE: {today}')
    print(f'|INFO| 제출 파일 저장 완료: {csv_name}')
    # print('하단 주소에ㅔ 접속하여 캐글Leaderboard 에 업로드 해주세요.')
    # print('https://www.kaggle.com/t/16531420f61345978c490712a7a5212b')

def parsing():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='bird/model_2.pth')
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parsing()
    net = torch.load(args.resume)
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0, 1]) # if multigpu
    test(net, args.batch_size)