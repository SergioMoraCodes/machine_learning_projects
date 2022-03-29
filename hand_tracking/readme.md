# Computer vision with python 
## Using mediapipe and cv2 libraries

it's important to create the basic instances to display the rest.

![](pine.png)

using a lot of images to explain what is going on, 

### creating tables
-tener en cuenta
-contenido
-para quien está escrito
-cual es el propósito

|shortcut | what it does|
|-|-|
|Ctrl+T|Show all symbols|
|Ctrl+Shift+Space|Trigger Parameters Hints|

```mermaid
flowchart TD;
    A[Input] --- |processing| B((Round));
    B ===> C{Decision};
    C -->|One| D[Result 1];
    C -.->|Three| E[Result 3];
    C -->|Two| f[Result 2];
    E -.-> |reorder|f[Result 2];
    D -->|finalprocess| g[Result 4];
    f -->|finalprocess| g[Result 4];
    g ===> |Feedback Loop| C
```

```mermaid
flowchart TD;
    A((Input)) & B((Input)) -->
    C((1)) & D((2)) & E((3)) & F((4));
    C & D & E & F --> 
    G((1)) & H((2)) & I((3)) & J((4));
```
