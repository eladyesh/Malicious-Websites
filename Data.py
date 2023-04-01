import re
import ssl
import socket
import whois21
import requests
from datetime import datetime, timedelta

IP_REGEX = re.compile(r'^(https?://)?([1-2]?\d{1,2}.){3}[1-2]?\d{1,2}')
HEX_IP_REGEX = re.compile(r'^(https?://)?(0x[\da-fA-F]{2}\.){3}(0x[\da-fA-F]{2})')
DOMAIN_REGEX = re.compile(r'^(https?://)?(www.)?(\w+\.)+\w+')
TINY_URL_LIST = ['t.co', 'goo.gl', 'bit.ly', 'amzn.to', 'tinyurl.com', 'ow.ly', 'youtu.be']
TINY_URL_REGEX = re.compile(fr'^(https?://)?(www.)?({"|".join(TINY_URL_LIST)})/')


def get_domain(url):
    for i in ('http://', 'https://'):
        if url.startswith(i):
            url = url[len(i):]
    if url.startswith('www.'):
        url = url[4:]
    return url.split('/')[0]


def add_prefix(url, https=True):
    if https:
        return 'https://www.' + url
    return 'http://www.' + url


class CollectData:

    @staticmethod
    def __getitem__(url: str):
        if not any((DOMAIN_REGEX.match(url), IP_REGEX.match(url), HEX_IP_REGEX.match(url))):
            return print('ERROR: url given is not formatted correctly.')
        if url.startswith('http://'):
            url = add_prefix(get_domain(url), False)
        url = add_prefix(get_domain(url))
        whois = CollectData.whois(get_domain(url))
        if whois is None: return print('ERROR: WhoIs failed.')
        print(' - Got WhoIs!')
        redirects, html, js = CollectData.get_source_code(url)
        print(' - Got Source Code!')
        data = []
        data.append(CollectData.check_ip_address(url));                 print(' 1 IP URL')
        data.append(CollectData.check_url_length(url));                 print(' 2 URL LEN')
        data.append(CollectData.check_tiny_url(url));                   print(' 3 TINY URL')
        data.append(CollectData.check_at(url));                         print(' 4 @')
        data.append(CollectData.check_redirect(url));                   print(' 5 //')
        data.append(CollectData.check_minus(url));                      print(' 6 -')
        data.append(CollectData.count_subdomain(url));                  print(' 7 SUBDOMAIN')
        data.append(CollectData.certificate(get_domain(url)));          print(' 8 CERTIFICATE')
        data.append(CollectData.domain_expiry(whois));                  print(' 9 DOMAIN EXPIRE')
        data.append(CollectData.check_favicon(url, html));              print('10 FAVICON')
        data.append(CollectData.https_token(url));                      print('11 TOKEN')
        data.append(CollectData.request_url(url, html));                print('12 REQUEST')
        data.append(CollectData.anchor_url(url, html));                 print('13 ANCHOR')
        data.append(CollectData.meta_script_link_url(url, html));       print('14 META SCRIPT LINK')
        data.append(CollectData.check_redirects(redirects));            print('15 REDIRECTS')
        data.append(CollectData.check_status_bar_change(html, js));     print('16 STATUS BAR')
        data.append(CollectData.check_right_click_disable(html, js));   print('17 RIGHT CLICK')
        data.append(CollectData.check_popup(html, js));                 print('18 POPUP')
        data.append(CollectData.check_iframe(html));                    print('19 IFRAME')
        data.append(CollectData.domain_age(whois));                     print('20 DOMAIN AGE')
        data.append(CollectData.dns_records(whois));                    print('21 DNS RECORDS')
        print('Finished Extracting Data!')
        return data

    @staticmethod
    def whois(url):
        for i in range(3):
            whois = whois21.WHOIS(url)
            if whois.success and whois.expires_date and whois.creation_date:
                return whois
        return None

    @staticmethod
    def get_source_code(url):
        response = requests.get(url, allow_redirects=True)
        redirects = len(response.history)
        html_source = response.text

        js_source = ''
        for js_url in re.findall(r'<script (.+ )?src="(.*?)"', html_source):
            if js_url[-1].startswith('/'): js_url = url + js_url[-1]
            else: js_url = js_url[-1]
            js_response = requests.get(js_url)
            js_source += js_response.text

        return redirects, html_source.lower(), js_source.lower()

    @staticmethod
    def check_ip_address(url):
        return 1 if IP_REGEX.match(url) or HEX_IP_REGEX.match(url) else -1

    @staticmethod
    def check_url_length(url):
        return -1 if len(url) < 54 else 0 if 54 <= len(url) <= 75 else 1

    @staticmethod
    def check_tiny_url(url):
        return 1 if TINY_URL_REGEX.match(url) else -1

    @staticmethod
    def check_at(url):
        return 1 if '@' in url else -1

    @staticmethod
    def check_redirect(url):
        return 1 if url.find('//', 8) > 0 else -1

    @staticmethod
    def check_minus(url):
        return 1 if '-' in url else -1

    @staticmethod
    def count_subdomain(url):
        return -1 if url.count('.') - 2 <= 1 else 0 if url.count('.') - 2 == 2 else 1

    @staticmethod
    def certificate(url):
        context = ssl.create_default_context()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with context.wrap_socket(sock, server_hostname=url) as conn:
            conn.connect((url, 443))
            cert = conn.getpeercert()
        expiry_date = datetime.strptime(cert['notAfter'], '%b  %d %H:%M:%S %Y GMT')
        return 1 if expiry_date - datetime.now() <= timedelta(days=365) else -1

    @staticmethod
    def domain_expiry(whois):
        return 1 if whois.expires_date - datetime.now() <= timedelta(days=365) else -1

    @staticmethod
    def check_favicon(url, html):
        link = re.search(r'<link(.|\n)*?>', html)
        if link is None: return -1
        link = link.group()
        link = link[link.find('href'):]
        if link.find('"') < 0 or 0 < link.find("'") < link.find('"'):
            href = link.split("'")[1]
        else:
            href = link.split('"')[1]
        return 1 if DOMAIN_REGEX.match(href) and get_domain(url) != get_domain(href) else -1

    @staticmethod
    def https_token(url):
        return 1 if 'https' in url else -1

    @staticmethod
    def request_url(url, html):
        images = re.finditer(r'<img(.|\n)*?>', html)
        outbound = 0
        total = 0
        for i in images:
            i = i.group()
            if 'href' in i:
                src = i[i.find('href'):]
                if src.find('"') < 0 or 0 < src.find("'") < src.find('"'):
                    src = src.split("'")[1]
                else:
                    src = src.split('"')[1]
                if DOMAIN_REGEX.match(src) and get_domain(url) != get_domain(src):
                    outbound += 1
                total += 1
            elif 'src' in i: total += 1
        if total == 0: return -1
        return -1 if outbound / total < 22 else 0 if 22 <= outbound / total <= 61 else 1

    @staticmethod
    def anchor_url(url, html):
        links = re.finditer(r'<a(.|\n)*?>', html)
        outbound = 0
        total = 0
        for i in links:
            i = i.group()
            for j in ('href', 'content'):
                if j in i:
                    src = i[i.find(j):]
                    if src.find('"') < 0 or 0 < src.find("'") < src.find('"'):
                        src = src.split("'")[1]
                    else:
                        src = src.split('"')[1]
                    if (DOMAIN_REGEX.match(src) and get_domain(url) != get_domain(src)) or not DOMAIN_REGEX.match(src):
                        outbound += 1
                    total += 1
            if 'src' in i: total += 1
        if total == 0: return -1
        return -1 if outbound / total < 31 else 0 if 31 <= outbound / total <= 67 else 1

    @staticmethod
    def meta_script_link_url(url, html):
        outbound = 0
        total = 0
        for i in ('meta', 'script', 'link'):
            links = re.finditer(fr'<{i}(.|\n)*?>', html)
            for i in links:
                i = i.group()
                if 'href' in i:
                    src = i[i.find('href'):]
                    if src.find('"') < 0 or 0 < src.find("'") < src.find('"'):
                        src = src.split("'")[1]
                    else:
                        src = src.split('"')[1]
                    if DOMAIN_REGEX.match(src) and get_domain(url) != get_domain(src):
                        outbound += 1
                    total += 1
                elif 'src' in i: total += 1
        if total == 0: return -1
        return -1 if outbound / total < 17 else 0 if 17 <= outbound / total <= 81 else 1

    @staticmethod
    def check_redirects(redirects):
        return -1 if redirects <= 1 else 0 if 2 <= redirects <= 4 else 1

    @staticmethod
    def check_status_bar_change(html, js):
        return 1 if (re.match(r'<script(.|\n)*?>', html) is not None and
                     re.search(r'onmouseover\s*=\s*([\'"])(.*?)\1',
                               re.search(r'<script(.|\n)*?>(.|\n)*?</script>', html).group()) and
                     'window.status' in re.search(r'onmouseover\s*=\s*([\'"])(.*?)\1',
                                                  re.search(r'<script(.|\n)*?>(.|\n)*?</script>', html).group()).group(2)) \
                    or \
                    (re.search(r'onmouseover\s*=\s*([\'"])(.*?)\1', js) and
                     'window.status' in re.search(r'onmouseover\s*=\s*([\'"])(.*?)\1', js).group(2)) \
                else -1

    @staticmethod
    def check_right_click_disable(html, js):
        return 1 if (re.match(r'<script(.|\n)*?>', html) is not None and
                     'event.button==2' in re.search(r'<script(.|\n)*?>(.|\n)*?</script>', html).group()) or \
                    'event.button==2' in js else -1

    @staticmethod
    def check_popup(html, js):
        return 1 if (re.match(r'<script(.|\n)*?>', html) is not None and
                     'window.prompt' in re.search(r'<script(.|\n)*?>(.|\n)*?</script>', html).group()) or \
                    'window.prompt' in js else -1

    @staticmethod
    def check_iframe(html):
        return -1 if re.match(r'<iframe(.|\n)*?>', html) is None else 1

    @staticmethod
    def domain_age(whois):
        return 1 if whois.creation_date - datetime.now() < timedelta(days=182) else -1

    @staticmethod
    def dns_records(whois):
        return -1 if whois.name_servers else 1