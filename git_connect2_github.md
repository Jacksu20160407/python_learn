# Git and Github

[Git和Github简单教程](http://www.cnblogs.com/schaepher/p/5561193.html#reset)
Git常用[命令](http://blog.csdn.net/dengsilinming/article/details/8000622)

# git and github
- **git add .** add to local

- **git commit -m "message for commit"**

- **git push -u origin master** this operation push the local repository to github(user_name and password needed))

  # connect local repository and remote github repository

  - create github repository in github.com 

  - create local git repository 

    ```python
    cd path/to/create/repository
    mkdir folder_named_repository
    cd folder_named_repository
    git init
    ```

  - connect local repository and remote repository

      ```python
      git remote add origin git@github.com:Jacksu20160407/DL_Note_Semantic_Segmantation.git
      git pull origin master
      ```

Congratulations!! You can rewrite something in local and push it to remote repository using above commands

[Git from the inside out](https://codewords.recurse.com/issues/two/git-from-the-inside-out)
