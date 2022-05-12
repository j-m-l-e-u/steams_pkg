from steams.models.attention import single_train_attention, loss_attention, evaluation_bytarget_attention, predict_attention

train_dict = {
    "ED_NW_att": single_train_attention,
    "multi_head_att": single_train_attention

}

loss_dict = {
    "ED_NW_att": loss_attention,
    "multi_head_att": loss_attention
}

eval_bytarget_dict = {
    "ED_NW_att": evaluation_bytarget_attention,
    "multi_head_att": evaluation_bytarget_attention
}

predict_dict = {
    "ED_NW_att": predict_attention,
    "multi_head_att": predict_attention
}
