// This dummy program enables us to test Proxy Apps without having an actual
// binary for it. It simply prints out the command line arguments, so we can
// verify everything is being passed in as it should.

#include <stdio.h>

int main(int argc, char *argv[])
{
    printf("Dummy app\n");
    for (size_t i = 0; i < argc; i++)
    {
        printf("%s ", argv[i]);
    }
    printf("\n");
}