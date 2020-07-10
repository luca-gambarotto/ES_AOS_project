# Embedded Systems - project
Repository for mandatory project of Embedded Systems course a.y. 2019/2020

***GAMBAROTTO Luca*** - Mat. *928094* - C.P. *10502632*

<p>This project is aimed at coding and building a complete sample application using the OpenCL implementation offered by Mango. Specifically the application developed is a porting of the OpenCL implementation of the Needleman-Wunsch algorithm proposed in the Rodinia Benchmark Suite. The development of the project, aside from the pure code drawing up, allowed me to dig into the technologies for parallel coding, both hardware and software side, dealing with topics that are not usually covered in mandatory courses at Politecnico di Milano.<p>

<p><b> Setup instructions </b></p>
To install the sample copy the *sample* folder in your mangolibs folders, overwriting all the exisitng files. The execute the following instruction to compile the application:

```bash
~$ cd mangolibs
~$ make samples
```

<p><b> Execution instructions </b></p>
To execute the sample move to the mango installation directory with the following command:

```bash
~$ cd MANGO_DIR/usr/bin
```

Then execute the application using:

```bash
~$ cd ./nw_opencl 16 5
```
where 16 is the length of the two sequences to be randomly generated and 5 is the penalty values. The length value must be a multiple of 16.
