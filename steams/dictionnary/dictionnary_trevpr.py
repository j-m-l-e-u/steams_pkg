from steams.models.attention import single_train_attention, loss_attention, evaluation_bytarget_attention, predict_attention

train_dict = {
    "MLP_NW_dist_att": single_train_attention,
    "MLP_NW_dist_2_att": single_train_attention,
    "multi_head_att": single_train_attention
}

loss_dict = {
    "MLP_NW_dist_att": loss_attention,
    "MLP_NW_dist_2_att": loss_attention,
    "multi_head_att": loss_attention
}

eval_bytarget_dict = {
    "MLP_NW_dist_att": evaluation_bytarget_attention,
    "MLP_NW_dist_2_att": evaluation_bytarget_attention,
    "multi_head_att": evaluation_bytarget_attention
}

predict_dict = {
    "MLP_NW_dist_att": predict_attention,
    "MLP_NW_dist_2_att": predict_attention,
    "multi_head_att": predict_attention
}
