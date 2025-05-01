limit up to 2,000 non-commercial API calls per month

DOCS
Getting Started
Signup and Login
To get started, you must register a Wolfram ID and sign in to the .

Obtaining an AppID
Click the "Get an AppID" button to get your first AppID button to start the app creation process. Give your application a name, a simple description and select which app type to register an AppID. Each application must have its own unique AppID.

Using the Short Answers API
Sample Query
Now that you have an AppID, you can make your first query. The base URL for queries is:

http://api.wolframalpha.com/v1/result
Every query requires two pieces of information—an AppID and an input value—in order to be processed correctly. The appid parameter tells your query which AppID to use:

http://api.wolframalpha.com/v1/result?appid=DEMO
Next, use the i parameter to specify the URL-encoded input for your query. For instance, here is a query for "How far is Los Angeles from New York?":

http://api.wolframalpha.com/v1/result?appid=DEMO&i=How+far+is+Los+Angeles+from+New+York%3f
When executed with a valid AppID, this URL will return a short line of text with a computed response to your query::

2464 miles
URL Parameters and Options
You can add URL-encoded parameters to customize output. Since the output for this API is plain text, only basic parameters are available.

units
Use this parameter to manually select what system of units to use for measurements and quantities (either "metric" or "imperial"). By default, the system will use your location to determine this setting. Adding "units=metric" to our sample query displays the resulting distance in kilometers instead of miles:

http://api.wolframalpha.com/v1/result?appid=DEMO&i=How+far+is+Los+Angeles+from+New+York%3f&units=metric
The result is now given in kilometers instead of miles:

3966 kilometers
timeout
This parameter specifies the maximum amount of time (in seconds) allowed to process a query, with a default value of "5". Although it is primarily used to optimize response times in applications, the timeout parameter may occasionally affect what value is returned by the Short Answers API.

Errors
HTTP Status 501
This status is returned if a given input value cannot be interpreted by this API. This is commonly caused by input that is misspelled, poorly formatted or otherwise unintelligible. Because this API is designed to return a single result, this message may appear if no sufficiently short result can be found. You may occasionally receive this status when requesting information on topics that are restricted or not covered.

HTTP Status 400
This status indicates that the API did not find an input parameter while parsing. In most cases, this can be fixed by checking that you have used the correct syntax for including the i parameter.

Invalid appid (Error 1)
This error is returned when a request contains an invalid option for the appid parameter. Double-check that your AppID is typed correctly and that your appid parameter is using the correct syntax.

Appid missing (Error 2)
This error is returned when a request does not contain any option for the appid parameter. Double-check that your AppID is typed correctly and that your appid parameter is using the correct syntax.