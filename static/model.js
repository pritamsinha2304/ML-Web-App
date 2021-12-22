$(document).ready(function(e) {
    $('.navTrigger').click(function () {
                $(this).toggleClass('active');
                console.log("Clicked menu");
                $("#mainListDiv").toggleClass("show_list");
                $("#mainListDiv").fadeIn();
    });

    $(window).scroll(function() {
        if ($(document).scrollTop() > 50) {
            $('.nav').addClass('affix');
            console.log("OK");
        } else {
            $('.nav').removeClass('affix');
        }
    });

    $("ul.navbar-nav > li").click(function (e) {
       $("ul.navbar-nav > li").removeClass("active");
       $(this).addClass("active");
    });

    window.setTimeout(function() {
        $('.message').fadeOut('slow');
    }, 5000);

    $("#predict").click(function(){
        if ($('#predict_file')[0].files.length == 0) {
            $('#predict').hide();
        }
    });

});