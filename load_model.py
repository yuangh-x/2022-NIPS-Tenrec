
def new_adj_stack(model, last_model, last_block_num, args):
    n = int(args.block_num / last_block_num)
    for i in range(last_block_num):
        if 'bert' in args.model_name:
            for j in range(n):
                model.bert.transformer_blocks[i * n + j] = \
                    last_model.bert.transformer_blocks[i]
        else:
            for j in range(n):
                model.residual_blocks[i * n + j] = last_model.residual_blocks[i]

    return model

