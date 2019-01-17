
let back_host = "127.0.0.1";
let back_port = "8000";


var show_flag = 0;
function know_you() {
    let comment_text = document.getElementsByClassName("send-comment-text")[0].value;
    console.log("懂你", comment_text);
    let get_score_url = "http://" + back_host + ":" + back_port + "/get_score?comment=" + comment_text;
    let comment_xml_http = new XMLHttpRequest();
    comment_xml_http.open('GET', get_score_url, true);
    comment_xml_http.send();
    comment_xml_http.onload = function () {
        let score = parseFloat(comment_xml_http.responseText);
        console.log("python获取到的分数", score);
        let double_score = parseInt(score * 2);
        let a = parseInt(double_score / 2);
        let b = double_score % 2;
        let c = 5 - a - b;
        let k = 1;
        let score_str = '<div class="block-stars"><p name="comment_score">' + score + '分</p><ul class="w3l-ratings">';
        for (let i = 0; i < a; i++) {
            score_str += '<li><a onclick="star_press(' + k + ')"><i name="star' + k + '" class="fa fa-star" aria-hidden="true"></i></a></li>\n';
            k++;
        }
        for (let i = 0; i < b; i++) {
            score_str += '<li><a onclick="star_press(' + k + ')"><i name="star' + k + '" class="fa fa-star-half-o" aria-hidden="true"></i></a></li>\n';
            k++;
        }
        for (let i = 0; i < c; i++) {
            score_str += '<li><a onclick="star_press(' + k + ')"><i name="star' + k + '" class="fa fa-star-o" aria-hidden="true"></i></a></li>\n';
            k++;
        }
        score_str += '</ul><input type="submit" value="懂你" onclick="know_you()"></div>\n';
        document.getElementsByName("comment_star")[0].innerHTML = score_str;
    }
}
function star_press(co) {
    console.log(co);
    for (let i = 1; i <= co; i++) {
        document.getElementsByName("star" + i)[0].className="fa fa-star";
    }
    for (let i = co + 1; i <= 5; i++) {
        document.getElementsByName("star" + i)[0].className="fa fa-star-o";
    }
    document.getElementsByName("comment_score")[0].innerText = co+"分";
}
function show_button_click() {
    // console.log("按钮点击");

    let media_list = document.getElementsByClassName("media");
    console.log(media_list);
    if (show_flag === 0) {
        // console.log("按钮文本",document.getElementsByClassName("show_button")[0].value)
        show_flag = 1;
        for (let i = 0; i < media_list.length; i++) {
            let comment_text = media_list[i].getElementsByClassName("media-body")[0].getElementsByTagName("p")[0].innerHTML;
            console.log("输入到python的评论", comment_text);
            let get_score_url = "http://" + back_host + ":" + back_port + "/get_score?comment=" + comment_text;
            let comment_xml_http = new XMLHttpRequest();
            comment_xml_http.open('GET', get_score_url, true);
            comment_xml_http.send();
            comment_xml_http.onload = function () {
                let score = parseFloat(comment_xml_http.responseText);
                console.log("python获取到的分数", score);
                let double_score = parseInt(score * 2);
                let a = parseInt(double_score / 2);
                let b = double_score % 2;
                let c = 5 - a - b;
                let ex_score_str = '<div class="block-stars"><p>' + score + '分（预测分数）</p><ul class="w3l-ratings">' + '<li><a href="#"><i class="fa fa-star" aria-hidden="true"></i></a></li>\n'.repeat(a) + '<li><a href="#"><i class="fa fa-star-half-o" aria-hidden="true"></i></a></li>\n'.repeat(b) + '<li><a href="#"><i class="fa fa-star-o" aria-hidden="true"></i></a></li>\n'.repeat(c) + '</ul></div>';
                media_list[i].innerHTML += ex_score_str;
            };
            document.getElementsByClassName("show_button")[0].value = "取消显示评估分数";
        }
    } else {
        show_flag = 0;
        for (let i = 0; i < media_list.length; i++) {
            console.log(media_list[i]);
            media_list[i].lastChild.remove();
        }
        document.getElementsByClassName("show_button").value = "显示评估分数";
    }

}

function send_button_click() {
    let comment_text = document.getElementsByClassName("send-comment-text")[0].value;
    console.log(comment_text);
    let container = document.getElementsByClassName('media-grids')[0];
    let comment = document.createElement('div');
    comment.className = 'media';
    let comment_score_text = document.getElementsByName("comment_score")[0].innerText;
    let comment_score = comment_score_text.substring(0, comment_score_text.lastIndexOf('分'));
    if (comment_score === '-') {
        comment_score = '3';
    }
    console.log("评论分数", comment_score_text, comment_score);
    comment.innerHTML = get_comment_item("Tony Pan", comment_text, comment_score);
    container.insertBefore(comment,container.firstChild);
    comment_text = document.getElementsByClassName("send-comment-text")[0].value = "";
}



function get_comment_item(username, comment, score) {
    let tot_str = '<h5>' + username + '</h5><div class="media-left"><a href="#"><img src="../images/user.jpg" title="One movies" alt=" " /></a></div><div class="media-body"><p>' + comment + '</p></div>';
    let double_score = parseInt(score * 2);
    let a = parseInt(double_score / 2);
    let b = double_score % 2;
    let c = 5 - a - b;
    console.log(a,b,c);
    let score_str = '<div class="block-stars"><p>' + score + '分</p><ul class="w3l-ratings">' + '<li><a href="#"><i class="fa fa-star" aria-hidden="true"></i></a></li>\n'.repeat(a) + '<li><a href="#"><i class="fa fa-star-half-o" aria-hidden="true"></i></a></li>\n'.repeat(b) + '<li><a href="#"><i class="fa fa-star-o" aria-hidden="true"></i></a></li>\n'.repeat(c) + '</ul></div>\n';
    // let ex_score_str = '<div class="block-stars"><p>' + score + '分（预测分数）</p><ul class="w3l-ratings">' + '<li><a href="#"><i class="fa fa-star" aria-hidden="true"></i></a></li>\n'.repeat(a) + '<li><a href="#"><i class="fa fa-star-half-o" aria-hidden="true"></i></a></li>\n'.repeat(b) + '<li><a href="#"><i class="fa fa-star-o" aria-hidden="true"></i></a></li>\n'.repeat(c) + '</ul></div>';
    return tot_str + score_str;
}


let movie_sql_id = parseInt(id) + 1;
let get_comment_url = "http://" + back_host + ":" + back_port + "/get_comment_list?movie_id=" + movie_sql_id + "&limit=10&offset=0";
let xml_http = new XMLHttpRequest();
xml_http.open('GET', get_comment_url, true);
xml_http.send();
xml_http.onload = function () {
    let json = JSON.parse(xml_http.responseText);
    console.log(json);
    var container = document.getElementsByClassName('media-grids')[0];
    console.log("评论区域", container);
    json.forEach(item => {
        console.log(item);
        let item_str = '<h5>TONY PAN</h5>\<div class="media-left"><a href="#"><img src="../images/user.jpg" title="One movies" alt=" " /></a></div><div class="media-body"><p>郭敬明的电影不好看</p></div>';
        let comment = document.createElement('div');
        comment.className = 'media';
        comment.innerHTML = get_comment_item(item.username, item.comment, item.star);
        container.append(comment);
    })
};
