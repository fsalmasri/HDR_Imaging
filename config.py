import argparse

def get_configs():
    parser = argparse.ArgumentParser(description='HDR Optim')

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')

    parser.add_argument('--train_ds_path', type=str, default=r'C:\Users\fedai\Desktop\mustafa_data\Training',
                        help='Training set folder path')
    parser.add_argument('--train_xsl_path', type=str, default=r'C:\Users\fedai\Desktop\mustafa_data\training.xlsx',
                        help='Pre-processed Training set xsl file path')

    parser.add_argument('--test_ds_path', type=str, default=r'C:\Users\fedai\Desktop\mustafa_data\Test',
                        help='Test set folder path')
    parser.add_argument('--test_xsl_path', type=str, default=r'C:\Users\fedai\Desktop\mustafa_data\test.xlsx',
                        help='Pre-processed Test set xsl file path')

    parser.add_argument('--insize', type=list, default=(224, 224), help='size of input image')
    parser.add_argument('--outsize', type=list, default=(224, 224), help='size of output image')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    return args