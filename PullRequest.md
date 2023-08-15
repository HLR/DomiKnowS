### Pull Request Tutorial for Contributing to DomiKnowS

If you wish to contribute to DomiKnowS, one of the best ways is to submit a pull request (PR) on GitHub. This tutorial will walk you through the process of creating a PR for DomiKnowS.

#### 1. **Set Up Your Environment**

- **Fork the Repository:** Navigate to the [DomiKnowS GitHub repository](https://github.com/HLR/DomiKnowS) and click the "Fork" button at the top right corner. This creates a copy of the project in your GitHub account.

- **Clone the Forked Repository:** Use Git to clone your forked repository to your local machine:
  ```
  git clone https://github.com/YourUsername/DomiKnowS.git
  ```

- **Add the Upstream Repository:** This helps in syncing your fork with the original repository:
  ```
  git remote add upstream https://github.com/HLR/DomiKnowS.git
  ```

#### 2. **Start with an Issue**

- Before creating a PR, check the 'Issues' tab in the original DomiKnowS repository to see if there's an existing issue related to your planned contribution.
  
- If relevant, comment on the issue expressing your interest in working on it. If there isn't an existing issue, consider creating one for better communication.

#### 3. **Create a New Branch**

- Always create a new branch for your changes. This keeps your fork's main branch free from your local edits and makes it easier to create clean PRs.
  ```
  git checkout -b nameofyourbranch
  ```

#### 4. **Make Your Changes**

- Modify or add the code as necessary in your local repository. Ensure you follow the project's coding standards and guidelines.

#### 5. **Commit and Push Your Changes**

- Commit your changes with a clear and concise commit message.
  ```
  git add .
  git commit -m "Your descriptive commit message"
  ```

- Push the changes to your forked repository.
  ```
  git push origin nameofyourbranch
  ```

#### 6. **Sync with Upstream**

Before submitting your PR, pull any recent changes from the upstream repository to ensure your branch is up to date.

```
git fetch upstream
git merge upstream/main
```

If there are any merge conflicts, you'll need to resolve them now.

#### 7. **Create the Pull Request**

- Navigate to your forked repository on GitHub.
  
- Click the "New pull request" button.

- Ensure the base fork points to `HLR/DomiKnowS` and the base branch is `main`. The head fork should be your repository, and the branch should be the one where you made changes.

- Fill out the PR template, detailing the changes you've made, the reason for them, and any issues they might be related to.

#### 8. **Respond to Feedback**

- After submitting the PR, maintainers or contributors might provide feedback. Engage in the discussion, and make any recommended changes to your PR as necessary. You can push additional changes to the same branch in your fork, and they'll automatically be added to your open PR.

#### 9. **Final Steps**

Once your PR has been approved, the maintainers will merge it into the main branch of DomiKnowS. You can then safely delete your feature branch.

---

Thank you for considering contributing to DomiKnowS! Your efforts help improve the framework for everyone. Remember to always check the project's CONTRIBUTING guidelines, if available, as there might be specific steps or practices unique to the project.