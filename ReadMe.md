<a id="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">Parts Of Speech Tagging</h3>

  <p align="center">
    Predicting Parts of Speech in Text File
    <br />
    <a href="https://github.com/Arnav6508/Parts-Of-Speech-Tagging"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Methodology</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this project, we try to predic the parts of speech in a text file for all words. This is super useful for NLP applications. Possible applications:
- Identifying named entities

- Speech recognition

- Coreference Resolution



### Methodology

The project uses Hidden Markov Models (HMM) to model the relationship between words and their respective parts of speech. An HMM assumes that:

- Word = Observable variables and POS Tags = Hidden States
- The machine can only perceive the observable variable (words) and these variables only depend on the hidden states (POS Tags)
- The current state only depends on the previous state
- Transition probabilities determine the likelihood of moving from one POS tag to another.
- Emission probabilities define the likelihood of a word being associated with a specific POS tag.

To efficiently determine the most likely sequence of POS tags for a given sentence, the Viterbi algorithm is employed. It uses dynamic programming to compute the best sequence of states (tags) for the given observations (words) while considering transition and emission probabilities.


<!-- GETTING STARTED -->
## Getting Started

To use the project locally, just utilise the inference function in main.py for testing on personal data, you can also create your own model by using build_model function in main.py


### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app._

1. Clone the repo
   ```sh
   git clone https://github.com/Arnav6508/Parts-Of-Speech-Tagging
   ```

2. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Email -  arnavgupta6508@gmail.com


<p align="right">(<a href="#readme-top">back to top</a>)</p>

