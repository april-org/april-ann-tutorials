#include <cstdio>
#include <cstdlib>

// TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  60000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte   ??               label
// ........
// xxxx     unsigned byte   ??               label

int main(int argc, char **argv)
{
  FILE *f = fopen(argv[1], "r");
  unsigned int magic, numitems;
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
  unsigned char *data = new unsigned char[numitems];
  fread(data, sizeof(unsigned char), numitems, f);
  fclose(f);
  for (int i=0; i<numitems; ++i)
    printf ("%d\n", data[i]);
  return 0;
}
