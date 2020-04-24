def load_args(args):
    if args.args_set == "social-lstm":
        args.encoder_h_dim = 64     #32
        args.decoder_h_dim = 32     #1
        args.emb_dim = 32           #32
        args.batch_size = 128       #128
        args.lr = 0.001
        args.wd = 0.0
        args.dropout = 0.3         #0.3
        args.model_type = "social-lstm"
        args.lstm_pool = True
        args.plot_name = "social-lstm"
        return args
    elif args.args_set == "lstm":
        args.encoder_h_dim = 64
        args.decoder_h_dim = 32
        args.emb_dim = 32
        args.batch_size = 128
        args.lr = 0.001
        args.wd = 0.0
        args.dropout = 0.3
        args.model_type = "lstm"
        args.lstm_pool = False
        args.plot_name = "lstm"
        return args

    else:
        print("no args_set found with this name!")
        quit()