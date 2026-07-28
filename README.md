# **MirrorMyManga**
MirrorMyManga is an attempt to "westernize" the reading format of Japanese Manga i.e. making Manga more approachable to western folks(the people who have difficulty reading a page from right to left)
And vice versa, making Comics more aproachable to Manga Otakus(Weebs) who have forgotten reading from
left to right.

**MirrorMyManaga does not intend to disrespect the art and culture of Manga and Comics.** It is simply an attempt to make apporaching Manga a bit easier, hopefully...

## **Results**:
### **Before:**
![Before](IdealBefore.png)

### **After:**
![After](IdealAfter.png)

### **Installation**:
Currently, there is only one way to use MirrorMyManga software:
1. Without Docker Image:
    a.  Fork the MirrorMyManga repository via CLI: 
    ```
        $gh repo fork https://github.com/lazyaxe/MirrorMyManga
    ```
    b. Fork it via GitHub's GUI
    c. Clone the repository:
    ```
        $git clone https://github.com/lazyaxe/MirrorMyManga
    ```

2. Docker Images (Work in progress)

### Usage:
There is currently only two way to use MirrorMyManga:
1 CLI (Command Line Interface):

        #inside the directory run command in terminal/cmd prompt:
        $python3 -m pip install -e .
        
        #to verify install
        $mirrormymanga --help

        #to run a simple conversion of Manga
        $mirrormymanga [input_path] [output_path] 

2. Programatically:
    You can access MirrorMyManga programatically via both `app.py` file or `./mirrormymanga/app.py` 
    I have kept both the class version and file tree version of MirrorMyManga both work equivalently in performance and give the same results

3. Web App (Coming Soon)

### **Limitations:**
1. **Onomotopoeia/SFX words:**
* Currently the Hero of this project, PaddleOCR lacks the ability to accurately and consistently detect onomotopoeia/SFX words such as "BOOM", "CRASH", "VROOM", "ZOOM" in both japanese and english.

For example:
![sfx](sfx.png)


2. **Non straight alignment text:** The text which are at an incline are treated properly which results in gibbrish text:
We gives unexpected results like:
![gibberish](gibbrish.png)


3. **Cover Pages:** I have not implemented a way to skip the processing of cover pages and non-panel pages as it's not huge issue right now.
Example
![cvrpgissue](coverPageIssue.png)

4. **Poor Performance without GPU(Nvidia)/XPU/TPU:** As the **PaddleOCR is heavily dependent on CUDA accelration**, the transformation of each panel can take upto 7s compared to 0.5s for a low-end Nvidia GPU.
The PaddleOCR GPU defaults to CPU mode if it fails to detect CUDA.


## **Upcoming Features/Improvements:**
1. Text overlay instead of pure text ROI manipulation.
2. Onomotopoeia/SFX text detection: I'm currently looking for a efficient and easy to way to detect and "remove"/inpaint the SFX text so I can flip these type of text effectively.
3. Translation of Manga/Comics: This improvement will be dependent on how the above two features work.