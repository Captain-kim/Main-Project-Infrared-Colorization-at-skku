if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["res18unet-S6A-RGB2RGB_iter_10000.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-root', type=str, default='VC24',
                        help='root dir of dataset ')
    parser.add_argument('-batch_size', type=int, default=5,
                        help='batch_size for training ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()

    main(args)
