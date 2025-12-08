
# Table of Contents

1.  [Creating ASCII Art](#orgb713f43)
    1.  [Requirements](#org3790a74)
    2.  [Executing the application](#org97b868b)
2.  [Members](#org481ab54)

This repository contain the codes, reports and results of our final
work for the Fundamentals of Image Processing (INF01046). The aim of
this repository is to reproduce the results from [this paper](https://ttwong12.github.io/papers/asciiart/asciiart.html).

   .\\.
    O/\` O~.
  .7L.7\`!..~.
 .(.\\[.. ]\\.\`\\
.( 7.  OL .~ .\*.\\\`^\\.,<. ..                 .

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">.&gt;&#x2026;.\T~ .~..)~\. `$.O*`[        .,&#x00ad;<sup>c</sup>``.)</td>
</tr>
</tbody>
</table>

	      .[ [ ).\`\\.c\`&#x2026;    .  .. &#x2026;\*,0><sup>c</sup>\`\`..,>\*c\`\`L..
	      0L0. .7\`.                    .   .\&#x00ad;<..     ( [
	      ] )\\<.                        .\\/..        oL.[
	      ]>\`.                           \`\\.\&#x00ad;\*c.    T ].
	     .[                               .\`$O0cc`  7..[
                      7`   .*cco.                         .**cc..>. 7.
                    0!.   /. 7...o.                        ). .>`. ].
               ..\>c.   .![ .[   .\                         0<..  O(
                .c=#\\,,0.L .o-=*`.  .         .``c^\.           .!                ....
            .O<**\\..  .\$)\\.   0L .\`.   \\  .<#,)<cc~\`[          ].             ,/\`./\`c\*\\,
.$^^.    ..`^\,..c--~/`        ` 0+...[   .[0.         |         .>*:`.        .`\
           .,-!`OOOO```:^\        o~<,..   ((   ^\*c`.].         |         0:              .).
            0L)0*-c```cccccc^^*.   )##<.   !\\.    .>*.          |       >$&#x2026;              .(
  O&#x2026;>.           O\*,..0#.       .\`\*<<\`\`~!.           ]     .!..\`..\`=.           .]
     <=.OL            ..o\*c<..\`^,.-\\======<<\O..     ]~]\` ..Oc.O:                 0.
     \`.o#O=<.               &#x2026;&#x2026;\`~O>~..     ..O\`\`o~==LO!.0. .>^.              .\\!.
	&#x2026;0/!\O,.                  .\`=..\`c~\\.      .\`<.+     ].      /   ccccc\`..
	.\*\`.  &#x2026;.\`^=\\\\..              .o\\.  .Oc\*\\. **c~.(].          >**[0~
       o\`0\`../.        ..\`\`   .\*-=\\.     .]\O.  O/$~,| .~!           \`>..
       \\\`.  .L ..                  .O^\\ .!cc\*<sup>c</sup>\`. o&#x2026;   O~.         0L

1.  0\\7.                             O.   0.      .)L        ]L

.c\`!     I                                o^\\/       . \`.       .(
O ].                                      0.         ).OO~.      .\\
  ]                                     .<sup>7</sup>.          O\L.!       .().
  .[ 0.                                0(              .. .I       ] ).\\
   O\*\o.                               .       c\\.         O[      . ]\*O[
     .(\`\\.I          L         .  .+             \`\`\\     \*. |        .  T
      ]..!.[        .[.        .[ oL    .[         ]      O\\)           T
      .\\  .\`.       O/.         O~].    7.         ]       .!          0[0[
       )    \`.7\\   \\ !            0.    [          )c[     .\`         0>\*(.
       O[    .o.\`=,O].           .\\|   O.            |     7.       .(..!.
  0^\`\`..[         7\`(             .L   \\.            \`    7.           ./.
 OL 7\`.cccc.      [].             ).O\*c\`..O\`             *.         ..*.
 ]. [. 7&#x2026;      0L<.          0.7./..                ..          0^\`.
 .\`~<<<(   *.    o.7. ,.      ./$L! .L  7`>.         .`.         .`)+
                      O^~\o,,0>-$*. 7.      O/\`O\`L ].  ! \`. ..,\\~<sup>c</sup>\`.\`~\\..\`\O\`\`&#x2026;
	     .   .\`\*\\\\[,,\\\\\*\`.    O>\*\\,0\\~^\`\`&#x2026;        ..O\`\`\`.                                  .


<a id="orgb713f43"></a>

# Creating ASCII Art


<a id="org3790a74"></a>

## Requirements

This code as created using `python 3.11.2`. We use different libraries
in order to achieve the aimed results. To make it reproducible we
created a python virtual environment. All the libraries used for
running the application are described in the `requirements.txt`. This
file can be used to install all the needed libraries to run the code
using the pip command:

    pip install -r requirements.txt


<a id="org97b868b"></a>

## Executing the application

The application takes command line arguments in order to create ASCII
Art from a image. A normal execution would use three arguments, the `path` (`-p`), that
indicates the path to the image you want to convert, the `ratio`
(`-r`), who is the color threshold for the image, and the `columns` (`-c`),
the number of columns for the output ASCII Art. You can see all the
options using:

    python ascii.py -h

An example of execution would be:

    python ascii.py --path ../image-tests/image2.jpg --ratio 0.2 -c 40

This makes a ASCII Art made of **40 columns** (text width) from the
`image2.jpg` taking in count only 0.2% of the colors. A good chose of
ration is important to correctly get the image vertices.

	      7)\_
	     ]. I
	    \_[  .\\
	   .!    OL
\\\\==~~~~~~~/.     )~~~~~~~==
.>\_.                      0/\`.
  ..O^\\.               \_-\`.
      .O^.           -\`.
	].           !
       \_(     \_,     .[

1.  \\\_-\`..\`^\\.  ]\_

o.\_-\`.      .O^\\.I
+\`.             O>L


<a id="org481ab54"></a>

# Members

-   Rayan Raddatz de Matos ([@rddtz](https://github.com/rddtz))
-   Eduardo Magnus Lazuta ([@MagnusLazuta](https://github.com/MagnusLazuta))

    			    .O.~\*cc\`&#x2026;&#x2026;\`\`c~\\.     ,\\\*cc\`\`\`\`\`\`\`o\*-\\\\,.
    		    .#7c\`O..                 O~..7\`.                ..O\`\`~.
    		  ](.0[                        0[                        .[\`~.
    		.<\\  7.                        0                          ).|+
    	       .!O. .(                         0                          .[\`.(
    	      O7O(  7.                         0                           ].\`O\\
    	     .(/!. .[                          0                           .[  \`.
    	    0(7].  7.                          0                            ]. .)OI
    	   .(]OL  .[                           0                            .\\  )!O\\
    	  .!0L(   T                            0                             ]. .\IO\\
    	 .!.!7.  .[                            0L                            .+ .\L\\0,
    	./.7].   !                              [                             0L \`\`OL7\\
           .7.)O(   oL                              [                              I  .)\`.)L
           7.].!.  .!                               [                              0L  .\\].).
          7.O(7.   ].                               [                               +   O\\$.).
         7..!0L   .(                                [                               O.   \`oo.o.
        ]..<.(    ]&#x2026;.,\\\\~\*\*\*==ccccc==\*~\\,.        [       .,<=-*=======\*\*~\\\\\\\\\\,.. ).   .\*\*.).
       o..\T*. .,00\&#x00ad;<sup>c</sup>\`\`\`.OO\\\\=#####\\\\\\,..($\*\\     [   O\c\`,\\>\*\*<sup>ccccc</sup>^^=.. .O\`\`c<sup>\*</sup>\*\\\\.   .$+ ^.
      .L 0(]..O,.\\\\~--**co\\$\\\\>===\*\*\*\*\*==#\\,\`$*#,    [  >.~$^$o#>======\0`00L\ccc^**=~\\\\,..O$..).
    O/~**:\`&#x2026;.    &#x2026;.,,.\\\\\\\\\\\\\*\*\*\*^^<sup>\*</sup>\*=#\\,OO.0&#x2026;.[.,(\`\`.,\\\\\&#x00ad;---``=###\\\\,,,,....       ...`o``\\~.

\\\*<ccc\`\`\`\`\`\`\`&#x2026;&#x2026;.                     .0cc\`&#x2026;&#x2026;.ccc..                   &#x2026;&#x2026;..OOO\`\`\`\`ccccc!<.
