# Create Autoencoders to embedd chess positions

### Creating a new Autoencoder
To create a new autoencoder follow the principe in modules/autoencoder/paper/. create a new Autoencoder class which extends the modules/autoencoder/BaseAutoencoder.


### Converting PGN or FEN to bitboards 
- use modules/PGN_reader to read (unique) FEN strings from a PGN file
- use modules/FEN_converter to convert a FEN to a bitboard

### visualizing t-SNE plots
when creating autoencoders based from BaseAutoencoder, you can use modules/visualization/TSNE_visualizer to create plots like _Autoencoder_Convolutional_normal_double_filters_tactics_test.html_ to have a portable (html file) visualization of chess positions.