#include <cstdio>
#include <cstdlib>

// TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  10000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte   ??               pixel
// 0017     unsigned byte   ??               pixel
// ........
// xxxx     unsigned byte   ??               pixel

// Pixels are organized row-wise. Pixel values are 0 to 255. 0 means
// background (white), 255 means foreground (black).

int main(int argc, char **argv)
{
  FILE *f = fopen(argv[1], "r");
  unsigned int magic, numitems, numrows, numcols;
  if (f == 0) {
    perror("JARL!");
    return 1;
  }
  unsigned char w[4];
  fread(w, sizeof(unsigned char), 4, f);
  fread(w, sizeof(unsigned char), 4, f);
  numitems = (((w[0] << 24) & 0xFF000000) |
	      ((w[1] << 16) & 0x00FF0000) |
	      ((w[2] <<  8) & 0x0000FF00) |
	      ( w[3]        & 0x000000FF));
  fread(w, sizeof(unsigned char), 4, f);
  numrows = (((w[0] << 24) & 0xFF000000) |
	     ((w[1] << 16) & 0x00FF0000) |
	     ((w[2] <<  8) & 0x0000FF00) |
	     ( w[3]        & 0x000000FF));
  fread(w, sizeof(unsigned char), 4, f);
  numcols = (((w[0] << 24) & 0xFF000000) |
	     ((w[1] << 16) & 0x00FF0000) |
	     ((w[2] <<  8) & 0x0000FF00) |
	     ( w[3]        & 0x000000FF));
  printf ("%d %d\nascii\n", numrows*numitems, numcols);
  unsigned char *data = new unsigned char[numrows*numcols];
  for (int i=0; i<numitems; ++i) {
    fread(data, sizeof(unsigned char), numrows*numcols, f);
    int p=0;
    for (int j=0; j<numrows; ++j) {
      for (int k=0; k<numcols; ++k) {
	printf ("%f ", data[p++]/(float)255);
      }
      printf ("\n");
    }
  }
  fclose(f);
  return 0;
}
