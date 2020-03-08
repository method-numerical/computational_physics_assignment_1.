#include<stdio.h>
#include<stdlib.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_blas.h>

int main()
{
  //input matrix elements
  double data[]={1.0,0.67,0.33,
	       0.45,1.0,0.55,
	       0.67,0.33,1.0};
  
  //converting data to matrix
  gsl_matrix_view A=gsl_matrix_view_array(data,3,3);

  //some additional requirement: permutation matrix, and signum function (-1)^s.
  gsl_permutation *p=gsl_permutation_alloc(3);
  int s;

  //gsl decompose A into elements of L,U.
  //L is lower triangular matrix + unit matrix (all diagonal =1),U is upper triangular matrix.
  //gsl displays lower triangular part of L and upper triangular part of U only.
  //gsl displays L,U elements(simultaneously) row-wise.
  gsl_linalg_LU_decomp(&A.matrix,p,&s);

  //view L,U.
  printf("\ngsl decomposed A into as follows...., these are non trivial elements of L,U(simultaneously) row-wise:\n");
  gsl_matrix_fprintf(stdout,&A.matrix,"%g");

  //to verify L*U=A.

  //let first define L matrix
  gsl_matrix *L=gsl_matrix_alloc(3,3);
  for (int i=0;i<=2;i++)
    {gsl_matrix_set(L,i,i,1);
     for (int j=i+1;j<=2;j++) gsl_matrix_set(L,i,j,0);
     for (int j=0;j<i;j++)
       	{double k=gsl_matrix_get(&A.matrix,i,j);
	  gsl_matrix_set(L,i,j,k);
	};
    };
  printf("\nL matrix (row-wise):\n");
  gsl_matrix_fprintf(stdout,L,"%g");

  //showing L in proper matrix format
  printf("which is nothing but:\n");
  
  for (int i=0;i<=2;i++)
    {for (int j=0;j<=2;j++) printf("%.2g\t",gsl_matrix_get(L,i,j));
     printf("\n");
    };
  
  
  //let now define U matrix
  gsl_matrix *U=gsl_matrix_alloc(3,3);
  for (int i=0;i<=2;i++)
    { for (int j=0;j<i;j++) gsl_matrix_set(U,i,j,0);
      for (int j=i;j<=2;j++)
	{double k=gsl_matrix_get(&A.matrix,i,j);
	 gsl_matrix_set(U,i,j,k);
	};     
    };
  printf("\nU matrix (row-wise):\n");
  gsl_matrix_fprintf(stdout,U,"%g");

  //showing U in proper matrix format
  printf("which is nothing but:\n");
  for (int i=0;i<=2;i++)
    {for (int j=0;j<=2;j++) printf("%g\t",gsl_matrix_get(U,i,j));
     printf("\n");
    };

  //now verifying L*U=A
  //gsl matrix multiplication function actually multiply L and U, element by element which is not actually matrix multiplication.
  //BLAS has actual matrix multiplication support
  //this is why initially gsl/gsl_blas.h was imported.
  gsl_matrix *Product=gsl_matrix_alloc(3,3);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,L,U,0.0,Product);
  printf("\nproduct of L and U is (row-wise):\n");
  gsl_matrix_fprintf(stdout,Product,"%g");

  //showing Product in proper matrix format
  printf("which is nothing but:\n");
  for (int i=0;i<=2;i++)
    {for (int j=0;j<=2;j++) printf("%g\t",gsl_matrix_get(Product,i,j));
     printf("\n");
    };

  printf("\nturns out that it is same as given matrix A.\n");

  return 0;
}
  
  
