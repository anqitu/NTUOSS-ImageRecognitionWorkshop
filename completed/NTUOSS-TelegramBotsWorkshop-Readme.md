# NTUOSS Telegram Bots Workshop

*by [Clarence Castillo](https://github.com/clarencecastillo) and [Steve Ye](https://github.com/HandsomeJeff) for NTU Open Source Society*

This workshop is based on [nickoala/telepot](https://github.com/nickoala/telepot) and assumes intermediate knowledge of Python.

**Disclaimer:** *This document is only meant to serve as a reference for the attendees of the workshop. It does not cover all the concepts or implementation details discussed during the actual workshop.*
___

### Workshop Details
**When**: Friday, 22 Sept 2017. 6:30 PM - 8:30 PM.</br>
**Where**: Theatre@TheNest, Innovation Centre, Nanyang Technological University</br>
**Who**: NTU Open Source Society

### Questions
Please raise your hand any time during the workshop or email your questions to [me](mailto:hello@clarencecastillo.me) or [Steve](mailto:yefan0072001@gmail.com) later.

### Errors
For errors, typos or suggestions, please do not hesitate to [post an issue](https://github.com/clarencecastillo/NTUOSS-TelegramBotsWorkshop/issues/new). Pull requests are very welcome! Thanks!
___

## Task 0 - Getting Started

#### 0.1 Introduction

<!-- TODO: write about this workshop -->
<!-- TODO: write about telegram -->

For this tutorial, we'll be creating a *CatBot* which allows users to
1. get random facts about cats,
2. get random cat pictures, and
3. talk to an actual cat *LIVE*!\*

#### 0.2 Initial Setup

1.  Download a text editor of your choice. I strongly recommend either:
    1.  [Atom](https://atom.io/)
    2.  [Sublime Text 3](http://www.sublimetext.com/3)
    3.  [Brackets](http://brackets.io/)
2.  Download [this project](https://github.com/clarencecastillo/NTUOSS-TelegramBotsWorkshop/archive/master.zip) and unzip the folder to your preferred workspace.
3.  Download [Telegram Desktop](https://desktop.telegram.org/) if you want easier access to quickly debug your bot.
4.  Download the following required Python 3.6.x packages:
    1.  [telepot](https://github.com/nickoala/telepot)
    2.  [requests](https://github.com/requests/requests)
    1.  [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/)
5.  If you don't already have these accounts, sign up for [Heroku](https://signup.heroku.com) and [Dropbox](https://db.tt/v1tjjEafrg)\**.
6.  Download Dropbox [desktop client](https://www.dropbox.com/install).

> **Note**<br>
> The easiest way to install python modules is via [pip](https://pypi.python.org/pypi/pip). Running this one-liner will install all of the required modules indicated above:
> <br>`pip install telepot requests beautifulsoup4`

\* *No cats were harmed in the making of this workshop.*
<br>\** *Get extra 500mb when you sign up using this link.*

## Task 1 - BotFather

#### 1.1 Name Your Cat Bot

Since we need unique names for our catbot, you can name your bot after your very own cat (or the cat you've never had). That way, BotFather won't likely complain about reserved usernames. If you don't have a cat, I've compiled a *Cat Name Matrix* aka "What's Your Cat Name" that could help you name your catbot:

**Step 1:** Sum of the digits in your Matriculation Number mod 10

| Number | Name | Number | Name |
| --- | --- | --- | --- |
| 0 | `Mister` | 5 | `Moon` |
| 1 | `Miss` | 6 | `Sir` |
| 2 | `Lord` | 7 | `Captain` |
| 3 | `Lady` | 8 | `General` |
| 4 | `Princess` | 9 | `Professor` |

<br>**Step 2:** First initial of your given name

| Letter | Name | Letter | Name |
| --- | --- | --- | --- |
| a | `Fluffy` | n | `Cuddles` |
| b | `Banana` | o | `Sparkly` |
| c | `Rainbow` | p | `Fatty` |
| d | `Mainbitch` | q | `Querty` |
| e | `Buddy` | r | `Hairy` |
| f | `Kissy` | s | `Sassy` |
| g | `Bitchy` | t | `Motherfucker` |
| h | `Kitty` | u | `Cutiepie` |
| i | `Handsome` | v | `Scratchy` |
| j | `Stormy` | w | `Moon` |
| k | `Pussy` | x | `Pouncey` |
| l | `Spiderpig` | y | `Lucky` |
| m | `Midnight` | z | `Snowy` |

<br>**Step 3:** Your GPA times your day of birth mod 10

| Digit | Name | Digit | Name |
| --- | --- | --- | --- |
| 0  | `Bellyrubs` | 5 | `Waggles` |
| 1 | `Purbox` | 6 | `Furface` |
| 2 | `Wiggles` | 7 | `Fluffles` |
| 3 | `Munchkin` | 8 | `Moon` |
| 4 | `Cheeks` | 9 | `Meowers` |

For example, if my matriculation number is U1359234X, my catbot's honorific would be `Captain` as derived from `(1+3+5+9+2+3+4)%10 = 7` . Combined with my name's first initial `C` (which maps to `Rainbow`) and given my *5.0* GPA times the day of my birth 20, `5.0*20%10 = 0` (which maps to `Bellyrubs`), my catbot's full name would be `Captain Rainbow Bellyrubs`. Consequently, my catbot's username would be `captainrainbowbellyrubscatbot`.

For you lazy bums out there, Steve built a bot that helps you figure out your catbot's name for you (see `examples\catfather.py`). Its name is `CatFather`. You can find it by its handle `@CatMotherBot` (because feminism?). Oh, and when you're done naming your catbot, you will be greeted with the opportunity help us track our task completion rate. Neat, huh?

*We will host `CatFather` for this session only.*

#### 1.2 Acquire Telegram HTTP API Token

Once we've got our catbot name settled, the next step is to acquire a token from Telegram. To do this, we need to let BotFather know that we want to create a bot. Open Telegram and search for *BotFather* in the contacts search field. Ask it to register a new bot for you by sending the command `/newbot` after which it'll ask you for your bot's name and username.

![task 1.2 screenshot](screenshots/task_1_2.png?raw=true)

Take note of our new bot's token to access Telegram's HTTP API as we'll need it in the code we're about to write so we can tell Telegram the behavior of our catbot.

The token should look something like this:

```
2351749591:VskzWEJdWj_rlCx23Hyu5mIJdWjTVskzdEx
```

#### 1.3 Meow! Hello World

![task 1.3 screenshot a](screenshots/task_1_3_a.png?raw=true)

Now that we got everything we need to start developing our bot, let's start programming to define the behavioral aspects of our catbot. To start off, open `catbot/catbot.py` on your text editor and paste the following snippet to where the indicated `TODO` is at (subsequent pasting of snippets will follow the same way):

```python
# TODO: 1.3.1 Create Hello World

# default response (feel free to change it)
response = 'Meow!'

# handle only messages with text content
if content_type == 'text':

    # get message payload
    msg_text = msg['text']

    # TODO: 2.1.1 Implement Command Handling #####################################
    response += ' Hello world!'

# send the response
bot.sendMessage(chat_id, response)
```

Don't forget to add in our bot's API token:
```python
# TODO: 1.3.2 Replace Token

TOKEN = '2351749591:VskzWEJdWj_rlCx23Hyu5mIJdWjTVskzdEx'
```

Let's bring our bot to life by running it. If you didn't get any errors this far, you should be able to test your bot by talking to it via Telegram Desktop. Enter your bot's username in the contacts search field and press `start` to begin the conversation. Remember that bots cannot initiate conversations with users so you have to send it a message first.

## Task 2 - Handling Message Events

#### 2.1 Commands

When processing a message, few pieces of information are presented which we will be using to determine the *intention* of the user. Much like instructions that you give to your cat (e.g. `sit`, `fetch` or `bark`), **commands** present a flexible yet semantic way to communicate with your bot. The following syntax may be used:

```
/command [optional] [argument]
```

As you may have noticed, commands start with `/` and may contain latin letters, numbers and underscores. We can suggest to the user what commands our bot understands by providing this information to BotFather. Doing so enables command suggestions where typing a `/` shows the user a list of available commands (more on this later).

For this section, we'll program our catbot such that it would be able to recognize and behave *differently* provided the following commands:

| Command | Description |
| --- | --- |
| `/ask` | This command returns a random fact about cats. |
| `/status` | This command returns the status of the cat together with a random picture of it. |
| `/feed` | This command returns the response of the cat after feeding it. |
| `/clean` | This command returns the response of the cat after bathing it. |
| `/kitty` | This command murders your current cat if it's still alive and spawns you a new kitten. |
| `/meow` | This command initiates an interactive conversation with the cat. |

Before we start, let's tell BotFather first of the commands we'll be writing. To do this, start by searching for `BotFather` and then type `/mybots` to get a list of your registered bots. Click the bot you just created and then click `Edit Bot` followed by `Edit Commands` button which should prompt you to type your bot's commands. To save us time, just paste the following snippet:

```
ask - Returns a random fact about cats
status - Returns the status of the cat together with a random picture of it
feed - Returns the response of the cat after feeding it
clean - Returns the response of  the cat after bathing it
kitty - Murders your current cat if it's still alive and spawns you a new kitten
meow - initiates an interactive conversation with the cat
```

For this tutorial, we will be using a *very sophisticated* cat simulator class which has already been coded for us. You're free to explore the codes inside `catbot/cat.py` for personal learning, but it would be beyond the scope of this workshop. For now, use the information below as reference of the commands made available to us to use when interfacing with an instance of the Cat class.

-   Attributes:
    -   `name (str)`: Name of cat
    -   `hunger (int)`: Tracks how hungry the cat is (-100 to 100)
    -   `dirt (int)`: Tracks how dirty the cat is (-100 to 100)
    -   `cycles_passed (int)`: Tracks how many cycles (updates) the cat has been through
    -   `is_alive (bool)`: Indicates whether the cat is still alive

-   Methods:
    -   `get_status()`: Returns the cat's age, needs and whether if it's still alive
    -   `feed()`: Feeds the cat (hunger - 25); returns the cat's response after feeding
    -   `clean()`: Cleans the cat (dirt - 25); returns the cat's response after cleaning
    -   `chat()`: Returns a hyper realistic cat response
    -   `kill()`: Resets the cat

The idea is to take care of a simulated cat using our bot. The cat's behavior and states have already been taken care of by `cat.py`, so the only thing left for us to do is to link the class' methods to our `on_chat_message` function. Our user would then be able to *take care* of the cat via the commands provided.

![task 2.1 screenshot a](screenshots/task_2_1_a.png?raw=true)

Let's update our code by adding routing logic which would direct the program flow according to the user's command. Replace everything below this `TODO 2.1.1` inside the same clause with the following:

```python
# TODO: 2.1.1 Implement Command Handling

if (msg_text.startswith('/')):

    # parse the command excluding the '/'
    command = msg_text[1:].lower()

    # prepare the correct response based on the given command
    if (command == 'ask'):
        # TODO: 3.1.3 Call Get Random Cat Fact ###############################
        response = 'Meow? *licks paws*'
    elif (command == 'status'):
        bot.sendMessage(chat_id, cat_bot.get_status())
        # TODO: 3.2.3 Send User Random Cat Image #############################
        return
    elif (command == 'feed'):
        response = cat_bot.feed()
    elif (command == 'clean'):
        response = cat_bot.clean()
    elif (command == 'kitty'):
        # TODO: 2.2.1 Confirm User Action Using Keyboard #####################

        # kill cat if still alive
        if (cat_bot.is_alive):
            bot.sendMessage(chat_id, 'Meow! *scratches your face*')
            bot.sendMessage(chat_id, cat_bot.name + ' was killed.')

        # respawn cat
        cat_bot.kill()
        bot.sendMessage(chat_id, '*respawns ' + cat_bot.name + '*')
    # TODO: 4.2.3 Handle 'meow' Command ######################################
    elif (command == 'meow'):
        response = 'Purrr~'

    # suggest the user to respawn the cat using /kitty
    if not (cat_bot.is_alive):
        response += ' You can respawn your cat using the command /kitty.'
else:
    # TODO: 4.2.4 Handle Conversation States #################################

    # talk to the cat if no command was matched
    response = cat_bot.chat()
```

Notice that we can actually send multiple messages to the user within the context of a single message event (see where we handle `/kitty`). Since telepot's default implementation is *synchronous*, the user will still receive the messages in the very same order we're sending them.

![task 2.1 screenshot b](screenshots/task_2_1_b.png?raw=true)

#### 2.2 Keyboards

Besides receiving, parsing and sending message back and forth, Telegram Bots allow richer interactions with the user using keyboards. To prevent our user from *accidentally* `/kitty`-ing our live cat, we can prompt the user again to confirm this action. To do this, let's add an **Inline Keyboard** which we could use to prompt the user to confirm his/her action. There's another type of keyboard which we'll be using in the later part of this tutorial.

![task 2.2 screenshot a](screenshots/task_2_2_a.png?raw=true)

Replace the previous implementation of handling `/kitty` with the following:

```python
# TODO: 2.2.1 Confirm User Action Using Keyboard

if (cat_bot.is_alive):

    # prepare confirm keyboard
    confirm_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text='Confirm', callback_data='kitty-confirm')],
        [InlineKeyboardButton(text='Cancel', callback_data='kitty-cancel')],
    ])

    # send response with keyboard
    response += ' Warning: ' + cat_bot.name + ' is still alive. Issuing this command will kill the cat (brutally) and reset all progress. Please confirm your action.'
    bot.sendMessage(chat_id, response, reply_markup = confirm_keyboard)

    # terminate prematurely to avoid sending message twice
    return

else:

    # respawn cat
    cat_bot.kill()
    bot.sendMessage(chat_id, '*respawns ' + cat_bot.name + '*')
```

While you're at it, don't forget to import `InlineKeyboardMarkup` and `InlineKeyboardButton` from `telepot.namedtuple` at the beginning of our code. This will load the necessary classes required to build our inline keyboard.

```python
# TODO: 2.2.2 Import Keyboards

from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
```

To handle the inline keyboard *on press* event, let's add some logic to our `on_callback_query`. This handler is to be called whenever the user presses any buttons on our custom keyboard. More specifically, this function is called whenever a callback query is initiated. Go ahead and paste the following code:

```python
# TODO: 2.2.3 Handle Callback Query

# close inline keyboard
inline_message_id = msg['message']['chat']['id'], msg['message']['message_id']
bot.editMessageReplyMarkup(inline_message_id, reply_markup=None)

# kill cat (brutally) on confirm
if (query_data == 'kitty-confirm'):
    bot.sendMessage(from_id, 'Meow!!!')
    bot.sendMessage(from_id, '*scratches your face*')
    bot.sendMessage(from_id, cat_bot.name + ' was killed.')

    #respawn cat
    cat_bot.kill()
    bot.sendMessage(from_id, '*respawns ' + cat_bot.name + '*')
else:
    bot.sendMessage(from_id, text='Meowww~~~')
    bot.sendMessage(from_id, text='*licks your face*')
```

![task 2.2 screenshot b](screenshots/task_2_2_b.png?raw=true)

Note that we've manually set the inline keyboard to hide itself when the user makes his/her selection. Leaving it on otherwise would leave a free *kill switch* lying around and I think that's not very healthy for our cat.

## Task 3 - Dynamic Content

#### 3.1 APIs

Now that we've finished most of our bot's basic functionalities, let's add some dynamic content to spice things up. I mean, wouldn't it be very nice to hear your cat say something smart for once - aside from that very mundane *meow* of course.

![task 3.1 screenshot](screenshots/task_3_1.png?raw=true)

APIs are application interfaces that use HTTP requests to query or update data. Most public databases usually expose an endpoint (aka API) free of charge, but others may require permission or a license to use. If you're looking for something free and easy, I recommend looking up [ProgrammableWeb](https://www.programmableweb.com/)'s catalog of APIs. I'm sure there's something over there you can use for your own projects.

For this step, we'll be using [Cat Facts API](https://catfact.ninja) to give us a random cat fact (obviously). Technically, there are several ways to implement this but for simplicity, we will be using the library `requests` to help us send an HTTP GET request.

In `catbot/catbot.py`, create a function `get_random_cat_fact()` which sends an HTTP GET request to [Cat Facts API](https://catfact.ninja), parses the response, and returns the cat fact as a string. Don't forget to import `requests` at the top of the file.

```python
# TODO: 3.1.1 Get Random Cat Fact

def get_random_cat_fact():
    response = requests.get('https://catfact.ninja/fact').json()
    return response['fact']
```

```python
# TODO: 3.1.2 Import Requests

import requests
```

Once that's done, we need to call this function inside our message handlers so our user can actually see the results. Back to where we did our command routing, replace the entire clause where `/ask` was previously handled with the following:

```python
# TODO: 3.1.3 Call Get Random Cat Fact

# prepare response with random cat fact if cat is alive
if (cat_bot.is_alive):
    response = 'Meow! ' + get_random_cat_fact()
else:
    response = 'This cat is currently unavailable.'
```

#### 3.2 Web Scraping

Sometimes the information we need may not be easily accessible via an API. For cases like this, we need to use `BeautifulSoup4` to parse an HTML page, traverse to the DOM element as specified by a selector, and then extract the attribute or text which is the exact data that we need.

One way we could improve our catbot is to send a random cat image every time the `/status` command is invoked. We could, of course, just manually save some cat pictures to a local folder somewhere, but that wouldn't really give us rich *dynamic* content because realistically, one can only have so much cat pictures in one computer.

![task 3.2 screenshot a](screenshots/task_3_2_a.png?raw=true)

There are probably hundreds of cat image collections out there (some even with APIs), but for this example, we'll be using [Cutestpaw](www.cutestpaw.com) because of how simple they've structured their DOM tree. Similar to the previous step, add the following snippets to `catbot/catbot.py`:

```python
# TODO: 3.2.1 Import Random and BeautifulSoup

import random
from bs4 import BeautifulSoup
```

```python
# TODO: 3.2.2 Get Random Cat Image URL

def get_random_cat_image_url():

    # get a random page number
    max_pages = 296
    random_page_number = random.randint(1, max_pages)

    # get the random page
    rand_cat_image_url = 'http://www.cutestpaw.com/tag/cats/page/' + str(random_page_number)
    response = requests.get(rand_cat_image_url)

    # load the page into bs4 and get random image
    soup = BeautifulSoup(response.text, 'html.parser')
    image_container = soup.find('div', {'id': 'photos'})
    images = image_container.find_all('img')
    random_image = random.choice(images)

    return random_image['src']
```

```python
# TODO: 3.2.3 Send User Random Cat Image

bot.sendChatAction(chat_id, 'upload_photo')
bot.sendPhoto(chat_id, get_random_cat_image_url())
```

Notice that for this example, we're also using `requests` to fetch the HTML source. Also, since scraping takes a few extra seconds to do its thing, we need to give the user some sort of feedback to let him/her know what's going on. Instead of sending a message along the lines of “Uploading image, please wait…”, Telegram API already provides us with `sendChatAction` which does exactly what we want. The code above uses `upload_photo` to indicate the type of action.

![task 3.2 screenshot b](screenshots/task_3_2_b.png?raw=true)

Good on you for making it this far! We're almost done but just a quick heads up: the following section describes advanced topics beyond the scope of what CZ1003 covers (for those taking that mod). Some parts may look very alien and may require a bit more time to understand, but hey, I'm sure you'll do *fine*.

## Task 4 - Conversations

#### 4.1 Bot Delegation

Normally, a bot is made to be used by more than one user. However, as it stands, our catbot spawns only a single instance of itself. That means Captain Rainbow Bellyrubs collectively treats everybody it talks to as the same damn person! The problem may not be obvious at first but you wouldn't want your users' messages to get mixed up, would you? No, that's not very *classy* (premature pun). This is why we have to create separate instances for each user.

To do this, we need to import a few modules that would help our catbot spawn multiple instances of itself. You don't really need to know what each import does but just know that `DelegatorBot` is the one that helps us keep separate conversations.

```python
# TODO: 4.1.1 Import DelegatorBot

from telepot import DelegatorBot
from telepot.delegate import pave_event_space, per_chat_id, create_open
```

This next part is a little tricky - we are gonna wrap all we've done into something called a `class`. Paste the following snippet and place all of the other functions we've written under this in the same level as `__init__` (by indentation).

```python
# TODO: 4.1.2 Wrap CatBot Class

class CatBot(telepot.helper.ChatHandler):

    def __init__(self, *args, **kwargs):
        super(CatBot, self).__init__(include_callback_query=True, *args, **kwargs)
        self.state = MEOW_CHOOSE_LANG
        self.language = None
        self.sentiment = 0
```

Now that all the functions we've written are members of `CatBot`, we need to insert an additional parameter `self` to each of these functions. This is an *Object-Oriented Programming* approach where we create an instance of this `CatBot` *object* for each user that converses with our bot. Each of these *objects* have the same starting attributes, and are independent of one another.

Go ahead and add the special parameter `self` as shown below. Note that this new parameter should precede all other parameters in the function signature should it already have others written.

```python
def get_random_cat_fact(self):
  ...

def get_random_cat_image_url(self):
  ...

def on_chat_message(self, msg):
  ...

def on_callback_query(self, msg):
  ...
```

For all calls to these functions, we also need to prepend it with `self.` as these functions now live under `CatBot`. In particular, `get_random_cat_fact()` as implemented in `TODO 3.1.3` should now be `self.get_random_cat_fact()` and `get_random_cat_image_url()` in `TODO 3.2.3` should now be `self.get_random_cat_image_url()`.

Now, just replace our previous bootstrap code with the new one below and let `DelegatorBot` do the rest of the heavy lifting.

```python
# TODO: 4.1.3 Implement DelegatorBot

bot = DelegatorBot(TOKEN, [
    pave_event_space()
    (per_chat_id(), create_open, CatBot, timeout=100)
])
MessageLoop(bot).run_as_thread()
```

That's it! Give yourself a pat on the back you clever cat you.

#### 4.2 State Machines

Often, people are looking for more than just a Q&A catbot. They are looking for a catbot  who will respond based on your responses to their responses (and so forth). A catbot with whom they might share a genuine conversation.

Okay, maybe it's not entirely genuine, but it's pretty close. We will use the idea of *state machines* to build a catbot who will actually listen and care.
*(Warning: catbot may not listen or care)*

The concept is pretty simple:
Let `MEOW_CHOOSE_LANG` be the so called "rest state". When you initiate a specific interaction with the catbot, it will move on to another state `MEOW_CONFIRM_LANG`.

At `MEOW_CONFIRM_LANG`, the catbot can do other specific set of tasks. Based on user input, the catbot can then move on to other states, where it can perform other sets of tasks specific to that given state.

To draw an image, all traffic lights follow a similar concept. In this example, let's define our rest state as `RED`. Then after a few seconds (the condition to be met), the machine transits into the next state which is `AMBER` and then finally `GREEN`. And then right after it reaches the final state, it will have to wait for a few seconds again before it *resets* itself back to `RED`.

In essence, the states that we define here are just 'stages' that are ordinal in nature (literally just integers) and because they are sequential, we can easily check if we've already passed a certain state by comparing the states as if they were just numbers to begin with.

Whew, I hope I hadn't bored you off yet. Let's do a bit of coding, shall we? First, let's define our states just before `CatBot` class declaration. The exact numeric values of the constants below do not really impose any significance, but keep in mind that they have to be in a non-arbitrary sequence.

```python
# TODO: 4.2.1 Set Conversation States

MEOW_CHOOSE_LANG = 0
MEOW_CONFIRM_LANG = 1
MEOW_LEAD_TOPIC = 2
MEOW_REACT_SENTIMENT = 3
```

The following steps will deal with the actual implementation of the dialog interaction between the user and our catbot. As we can't possibly handle all use case scenarios, we'd need to use a keyboard to restrict the inputs the user can send our catbot. Import the other type of custom keyboard `ReplyKeyboardMarkup`, it's button class `KeyboardButton` and the `dialog` dictionary from `conversation.py` as follows.

```python
# TODO: 4.2.2 Import KeyboardButton, ReplyKeyboardMarkup and Conversation Dialog

from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton
from conversation import dialog
```

Now that we have defined our states and that our dialog is ready, replace the current handler of `/meow` with the following snippet just after the clause for the `/kitty` command.

```python
# TODO: 4.2.3 Handle 'meow' Command

elif (command == 'meow' and self.state == MEOW_CHOOSE_LANG):

    # get available languages
    languages = dialog.keys()

    # prepare custom keyboard
    choose_lang_keyboard = ReplyKeyboardMarkup(one_time_keyboard=True, keyboard=[
        [KeyboardButton(text=language)] for language in languages
    ] + [[KeyboardButton(text='Nevermind')]])

    response = 'Meow meow mrowww hisss'
    bot.sendMessage(chat_id, response, reply_markup=choose_lang_keyboard)

    # move to next state
    self.state = MEOW_CONFIRM_LANG

    # terminate prematurely to avoid sending message twice
    return
```

![task 4.2 screenshot a](screenshots/task_4_2_a.png?raw=true)

Notice that `ReplyKeyboardMarkup`, unlike it's cousin `InlineKeyboardMarkup`, does not issue a callback query on keyboard button *press*. This means that data on click will be sent to our bot though as if the user typed and sent that exact message.

Also, as you can see, if a user sends a `/meow` command at state `MEOW_CHOOSE_LANG`, the bot will transit into `MEOW_CONFIRM_LANG`, and a custom keyboard is generated to help the user further the conversation, in English, Spanish, or German! Now that is one multilingual cat!

Now, let's alter more code such that we can insert additional states. Replace the clause in `TODO 4.2.4` with this segment here:

```python
# TODO: 4.2.4 Handle Conversation States

# separates responses into before and after 'speak' has been called
if (self.state > MEOW_CHOOSE_LANG):

    # dialog variable placeholders
    dialog_keyboard = None
    dialog_response = ''

    if (self.state == MEOW_CONFIRM_LANG):
        # TODO: 4.2.5 Handle State MEOW_CONFIRM_LANG #####################
        pass

    elif (self.state == MEOW_LEAD_TOPIC):
        # TODO: 4.2.6 Handle State MEOW_LEAD_TOPIC #######################
        pass

    elif (self.state == MEOW_REACT_SENTIMENT):
        # TODO: 4.2.7 Handle State MEOW_REACT_SENTIMENT ##################
        pass

    bot.sendMessage(chat_id, dialog_response, reply_markup=dialog_keyboard)
    return

else:
    # talk to the cat if no command was matched and 'meow' not initialized
    response = cat_bot.chat()
```
Now, for state `MEOW_CONFIRM_LANG`, we'll make catbot give a warm greeting based on the language the user chose in the previous state `MEOW_CHOOSE_LANG`. We'll use a **ReplyKeyboardMarkup** again to let the user continue or end the conversation.

```python
# TODO: 4.2.5 Handle State MEOW_CONFIRM_LANG

if (msg_text != "Nevermind"):

    # update context language for later re-use
    self.language = msg_text

    # load language dialog
    language_dialog = dialog[self.language]

    # prepare greeting and next keyboard
    dialog_keyboard = ReplyKeyboardMarkup(one_time_keyboard=True, keyboard=[
        [KeyboardButton(text=language_dialog['yes'])], [KeyboardButton(text=language_dialog['no'])]
    ])
    dialog_response = language_dialog['greeting']  

    # move to next state
    self.state = MEOW_LEAD_TOPIC

else:

    # reset state when user cancels
    self.state = MEOW_CHOOSE_LANG
    dialog_response = 'Perhaps it\'s best to not think about it, eh?'
```

Now, if the user chooses to go along with the "conversation", our state machine will eventually transit into `MEOW_LEAD_TOPIC`. This is where a big chunk of `conversation.py` with all its lists of text comes in. I'm sorry, but this is basically how you fill the vocabulary of a chat bot. It's the ugly truth no one wants to hear.

Catbot will ask a random question in the form of a random leading clause + random topic, with a custom keyboard of the available answers. Remember to choose wisely! For there is actually a correct answer to each question.

```python
# TODO: 4.2.6 Handle State MEOW_LEAD_TOPIC

# load language dialog
language_dialog = dialog[self.language]

# get random sentiment
self.sentiment = random.randrange(2)

if (msg_text == language_dialog['yes']):

    # prepare response and next keyboard
    dialog_keyboard = ReplyKeyboardMarkup(one_time_keyboard=True, keyboard=[
        [KeyboardButton(text=sentiment)] for sentiment in language_dialog['sentiments'].values()
    ])

    # get a random leading clause
    dialog_response = random.choice(language_dialog['responses']['leading'])

    # get a random topic
    dialog_response += random.choice(sum(language_dialog['topics'].values(), [])) + '?'

    # move to next state
    self.state = MEOW_REACT_SENTIMENT

else:

    # reset state when user cancels
    self.state = MEOW_CHOOSE_LANG
    dialog_response = 'That\'s a shame.'
```

Next, assuming the user hasn't quit the conversation yet, our machine moves into the next state `MEOW_REACT_SENTIMENT`. Here, catbot responds based on the user's previous answer. For example, catbot asked about "Hitler" and the answer is "I love it!", then catbot will be angry, and respond with a random insult. *This goes for real life - don't go around telling people you love Hitler. It won't end well for you I tell ya.*

```python
# TODO: 4.2.7 Handle State MEOW_REACT_SENTIMENT

# load language dialog
language_dialog = dialog[self.language]

if (msg_text != language_dialog['sentiments']['cancel']):

    # get numeric value of user answer
    answer = [language_dialog['sentiments']['good'],
    language_dialog['sentiments']['meh'],
    language_dialog['sentiments']['bad']].index(msg_text)

    # prepare random response accordingly if user's answer matches bot's sentiment
    responses = language_dialog['responses']['good' if self.sentiment == answer else 'bad']
    dialog_response = random.choice(responses)

else:

    # prepare different response when user cancels
    dialog_response = "Was it all a dream? You could have sworn that cat just spoke to you..."

# reset state either way
self.state = MEOW_CHOOSE_LANG
```

Thus ends this "conversation" and the state reverts back to `MEOW_CHOOSE_LANG`. Of course, you can always insert more states to extend the feeling of actually talking to someone, but we're going to leave that part to you.

![task 4.2 screenshot b](screenshots/task_4_2_b.png?raw=true)

This is it for interacting with saved states. The full, working code may be found under `examples/multi_catbot.py`.

## Task 5 - Deployment

The final step is to free our catbot into the wild, that is, to get it deployed on a server. Of course there's always the option to keep your laptop or device running, but why do that when you can get it up and running on a remote server *for free*?

To do this, login to your Heroku dashboard ([sign up here](https://signup.heroku.com) if you still don't have one) and create a new app like this:

![task 5 screenshot a](screenshots/task_5_a.png?raw=true)

There's no specific naming convention for catbot deployment applications, but for this instance, let's name our app after our catbot like this: `captain-rainbow-bellyrubs-bot` (use your catbot's own name you dummy). Note that by default, free accounts only get access to US and Europe servers but it shouldn't be a problem for us whichever region we choose.

Once that's done, head over to *Deployment* in the *Deploy* tab of your new app and link your Dropbox  account ([sign up here](https://db.tt/v1tjjEafrg) if you still don't have one).

As you can see, there are other options for us to choose, but to make thing simple, we're only covering the Dropbox method for this workshop. I recommend you also try the *git* method - you'll learn a lot more useful stuff that way.

If things go as planned, you should see something like the screenshot below after linking your Dropbox account.

![task 5 screenshot b](screenshots/task_5_b.png?raw=true)

Next, we need to copy all of our files into the directory specified in the previous step: `Dropbox/Apps/Heroku/captain-rainbow-bellyrubs-bot`. This one shouldn't be that tough since you've already installed and connected your Dropbox account to your laptop. Just hit the *Deploy* button once all your files are uploaded.

![task 5 screenshot c](screenshots/task_5_c.png?raw=true)

Finally, to actually get our bot running, we need to tell Heroku to give us a free *dyno* (something like a processor). Go to *Free Dynos* under the *Resources* tab and enable the toggle button where it says `worker python catbot/catbot.py`. It should look something like this assuming you've followed the steps correctly.

![task 5 screenshot d](screenshots/task_5_d.png?raw=true)

___

## Acknowledgements

Many thanks to [anqitu](https://github.com/anqitu) for carefully testing this walkthrough and to everybody else in [NTU Open Source Society](https://github.com/ntuoss) committee for making this happen! :kissing_heart::kissing_heart::kissing_heart:

## Resources

[Telepot Docs](http://telepot.readthedocs.io/en/latest/#)
<br>[Telegram Bot API Docs](https://core.telegram.org/bots/api)
<br>[NTU CampusBot](https://github.com/clarencecastillo/ntu-campusbot)
