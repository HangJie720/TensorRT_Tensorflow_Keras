# plot accuracy & loss
plot_history(history)

# visualize model
plot_model(model, to_file="./Resources/model.png", show_shapes=True)

# save model & weight
json_string = model.to_json()
open('./Resources/model.json', 'w').write(json_string)

model.save_weights('./Resources/model_weights.hdf5')